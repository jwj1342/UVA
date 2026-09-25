#!/usr/bin/env python3
"""Disclosure per policy: Ask%, L_P1, OCO and the consultation accounting.

For every transmitted consultation, r_h is the local context observed before it (located by its
position in the message list), I(r_h) is built from it, and L_P1 / OCO of the SENT text are counted
against it; means are pooled over consultations. With --tb-tasks-dir the reference is instead each
Terminal-Bench task's environment definition and per-task inventory. Ask% is over the assigned
rollouts (--pool x --samples). Also: the detector-only count, NCVL = L_P1 / |I(r)|, --oco-n, and
per-task calls, blocked calls, empty replies, delivery failures, expert tokens and latency.

    python -m uva.eval.privacy_table --pool data/pools/verified_500.jsonl --samples 3 \
        --arm natural=output/runs/verified_natural_k0,output/runs/verified_natural_k1,output/runs/verified_natural_k2 \
        --out output/reports/privacy_verified.json
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

from uva.privacy import leakage as L


def _run_dirs(spec: str) -> list[str]:
    out = []
    for part in spec.split(","):
        out.extend(sorted(glob.glob(part)) or [part])
    return [d for d in out if Path(d).is_dir()]


def _q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else None


def arm_privacy(spec: str, n_assigned: int | None = None, oco_n: int = 5, tb_inventories=None) -> dict:
    ref_counts, det_counts, ncvl, oco_counts, inv_sizes = [], [], [], [], []
    lat, tin, tout = [], [], []
    inst_total = inst_asked = n_unlocated = 0
    calls_total = blocked = failed = empty = 0
    traj_files = [t for d in _run_dirs(spec) for t in sorted(glob.glob(f"{d}/*/*.traj.json"))]
    for t in traj_files:
        try:
            j = json.load(open(t))
        except (OSError, json.JSONDecodeError):
            continue
        iid = Path(t).parent.name
        calls = (j.get("info", {}).get("ask_expert") or {}).get("calls", [])
        inst_total += 1
        calls_total += len(calls)
        blocked += sum(1 for c in calls if c.get("outcome") in ("over_budget", "suppressed"))
        failed += sum(1 for c in calls if str(c.get("outcome", "")).startswith("failed") or c.get("outcome") == "misconfigured")
        empty += sum(1 for c in calls if c.get("outcome") == "empty_reply")
        transmitted = [(k, c) for k, c in enumerate(calls) if L.is_transmitted(c)]
        inst_asked += int(bool(transmitted))
        for k, c in transmitted:
            sent = L.sent_text(c)
            if tb_inventories is not None:
                inv, ctx = tb_inventories.get(iid, (set(), ""))
            else:
                ctx = L.consultation_reference(j, k)
                if ctx is None:  # not locatable in the message list; fall back to the step count
                    n_unlocated += 1
                    ctx = L.local_context(j, before_step=c.get("step"))
                inv = L.reference_inventory(ctx)
            inv_sizes.append(len(inv))
            n_ref = L.leaked_entities_ref(sent, inv)[0]
            ref_counts.append(n_ref)
            ncvl.append(n_ref / len(inv) if inv else 0.0)
            det_counts.append(L.leaked_entities(sent)[0])
            oco_counts.append(L.oco_ref(sent, ctx, n=oco_n)[0])
            if c.get("latency_s") is not None:
                lat.append(c["latency_s"])
            if c.get("expert_input_tokens") is not None:
                tin.append(c["expert_input_tokens"]); tout.append(c.get("expert_output_tokens") or 0)
        del j
    denom = n_assigned or inst_total
    n = len(ref_counts)
    row = {"n_q": n, "n_rollouts_found": inst_total, "n_rollouts_assigned": denom,
           "n_rollouts_asked": inst_asked,
           "ask_pct": round(100 * inst_asked / denom, 1) if denom else None,
           "consultations_not_located": n_unlocated,
           "accounting": {"calls_per_task": round(calls_total / denom, 3) if denom else None,
                          "blocked_calls_per_task": round(blocked / denom, 3) if denom else None,
                          "delivery_failures": failed, "empty_replies": empty,
                          "expert_input_tokens_mean": round(statistics.mean(tin), 1) if tin else None,
                          "expert_output_tokens_mean": round(statistics.mean(tout), 1) if tout else None,
                          "latency_s_median": _q(lat, 0.5), "latency_s_q95": _q(lat, 0.95)}}
    if not n:
        return row
    row.update({
        "calls_per_rollout_transmitted": round(n / denom, 3),
        "L_P1_mean": round(statistics.mean(ref_counts), 2),
        "L_P1_median": statistics.median(ref_counts),
        "L_P1_detector_only_mean": round(statistics.mean(det_counts), 2),
        "NCVL_mean": round(statistics.mean(ncvl), 4),
        "pct_calls_with_any_leak": round(100 * sum(1 for x in ref_counts if x > 0) / n, 1),
        f"OCO{oco_n}_mean": round(statistics.mean(oco_counts), 2),
        f"OCO{oco_n}_median": statistics.median(oco_counts),
        "pct_calls_with_code_overlap": round(100 * sum(1 for x in oco_counts if x > 0) / n, 1),
        "inventory_size_median": statistics.median(inv_sizes),
    })
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", default=[], help="NAME=run_dir_or_glob[,more...]")
    ap.add_argument("--pool", help="assigned instances (jsonl or id list): Ask% denominator = its size x samples")
    ap.add_argument("--samples", type=int, default=1, help="samples per instance in the arm (with --pool)")
    ap.add_argument("--oco-n", type=int, default=5)
    ap.add_argument("--tb-tasks-dir", help="Terminal-Bench: score against each task's environment-definition inventory")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n_assigned = None
    if args.pool:
        n_assigned = args.samples * sum(1 for l in Path(args.pool).read_text().splitlines() if l.strip())
    tb_inv = None
    if args.tb_tasks_dir:
        from uva.eval.terminalbench.inventory import load_inventories
        tb_inv = load_inventories(Path(args.tb_tasks_dir))
    rows = {}
    for spec in args.arm:
        name, pat = spec.split("=", 1)
        rows[name] = r = arm_privacy(pat, n_assigned, args.oco_n, tb_inv)
        print(f"{name}: rollouts={r['n_rollouts_found']}/{r['n_rollouts_assigned']} calls={r['n_q']} "
              f"Ask%={r['ask_pct']} L_P1={r.get('L_P1_mean')} (detector-only {r.get('L_P1_detector_only_mean')}) "
              f"OCO{args.oco_n}={r.get(f'OCO{args.oco_n}_mean')}"
              + (f"  [{r['consultations_not_located']} consultations located by step count]"
                 if r["consultations_not_located"] else ""))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
