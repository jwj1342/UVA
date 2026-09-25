#!/usr/bin/env python3
"""The master set of accepted pairs and the collection funnel, one batch at a time.

    python -m uva.data.pairs_master append --batch b0 --collect-dir output/runs/b0_collect \
        --candidates output/pairs/b0_candidates.jsonl --verified output/pairs/b0_verified.jsonl
    python -m uva.data.pairs_master report        # funnel; acceptance by repository, step, length
    python -m uva.data.pairs_master export --runs-root output/runs --out output/release/pairs.jsonl \
        --solve-steps-out output/release/solve_steps.jsonl      # rollout-free form for release

A candidate is a training pair when its q+ replay resolved the task (status `accepted` or
`uninformative_prefix_solved`); `accepted` also means the no-ask control failed, kept as the
`load_bearing` flag. `export` re-checks every clause of the gate from the rollout and attaches
each pair's history h (`prompt`) and disclosure counts.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

TRAIN_OK = {"accepted", "uninformative_prefix_solved"}


def _cid(c: dict) -> str:
    return f"{c['instance_id']}__{c.get('traj_sha16', '')}__{int(c.get('call_index', 0))}"


def _collect_stats(collect_dir: Path) -> dict:
    ev_path = collect_dir / "eval_results.json"
    ev = json.loads(ev_path.read_text()) if ev_path.exists() else {}
    resolved = {k for k, v in ev.items() if isinstance(v, dict) and v.get("resolved") is True}
    n = n_calls = n_answered = n_solved_calls = 0
    exits = collections.Counter()
    for t in glob.glob(str(collect_dir / "*" / "*.traj.json")):
        d = json.loads(open(t).read()); n += 1
        iid = d.get("instance_id") or Path(t).parent.name
        exits[str((d.get("info") or {}).get("exit_status"))] += 1
        calls = ((d.get("info") or {}).get("ask_expert") or {}).get("calls", [])
        n_calls += len(calls)
        ans = sum(1 for c in calls if c.get("outcome") == "answered")
        n_answered += ans
        if iid in resolved:
            n_solved_calls += ans
    return {"trajectories": n, "exit_status": dict(exits), "solved": len(resolved),
            "consultations_total": n_calls, "consultations_answered": n_answered,
            "consultations_in_solved_trajectories": n_solved_calls}


def append(args) -> None:
    pairs_dir = Path(args.pairs_dir); pairs_dir.mkdir(parents=True, exist_ok=True)
    master_path, funnel_path = pairs_dir / "master_pairs.jsonl", pairs_dir / "funnel.json"
    master = {}
    if master_path.exists():
        for l in master_path.read_text().splitlines():
            if l.strip():
                r = json.loads(l); master[_cid(r)] = r
    before = len(master)
    verified = [json.loads(l) for l in Path(args.verified).read_text().splitlines() if l.strip()]
    cands = [json.loads(l) for l in Path(args.candidates).read_text().splitlines() if l.strip()] \
        if Path(args.candidates).exists() else []
    rej_path = Path(args.candidates).with_suffix(".rejected.jsonl")
    rejected = [json.loads(l) for l in rej_path.read_text().splitlines() if l.strip()] if rej_path.exists() else []
    status = collections.Counter(r.get("status") for r in verified)
    by_k = collections.defaultdict(collections.Counter)
    added = 0
    for r in verified:
        by_k[int(r.get("call_index", 0))][r.get("status")] += 1
        if r.get("status") in TRAIN_OK:
            r = dict(r); r["batch"] = args.batch; r["load_bearing"] = r.get("status") == "accepted"
            if _cid(r) not in master:
                master[_cid(r)] = r; added += 1
    with master_path.open("w") as fh:
        for r in master.values():
            fh.write(json.dumps(r) + "\n")
    funnel = json.loads(funnel_path.read_text()) if funnel_path.exists() else {"batches": {}}
    funnel["batches"][args.batch] = {
        "collection": _collect_stats(Path(args.collect_dir)),
        "pre_replay_rejections": dict(collections.Counter(r.get("reject_stage") for r in rejected)),
        "candidates": len(cands),
        "candidates_by_call_index": dict(collections.Counter(int(c.get("call_index", 0)) for c in cands)),
        "replay_status": dict(status),
        "replay_status_by_call_index": {str(k): dict(v) for k, v in sorted(by_k.items())},
        "training_pairs": sum(status[s] for s in TRAIN_OK), "new_master_pairs": added,
    }
    funnel["master_pairs"] = len(master)
    funnel_path.write_text(json.dumps(funnel, indent=2))
    print(f"[master] batch {args.batch}: +{added} pairs ({before} -> {len(master)}); replay status {dict(status)}")


def report(args) -> None:
    funnel_path = Path(args.pairs_dir) / "funnel.json"
    funnel = json.loads(funnel_path.read_text()) if funnel_path.exists() else {"batches": {}, "master_pairs": 0}
    tot = collections.Counter(); st = collections.Counter(); pre = collections.Counter()
    k_st = collections.defaultdict(collections.Counter)
    for b, v in funnel["batches"].items():
        c = v["collection"]
        tot["trajectories"] += c["trajectories"]; tot["solved"] += c["solved"]
        tot["consultations_answered"] += c["consultations_answered"]
        tot["consultations_in_solved"] += c["consultations_in_solved_trajectories"]
        tot["candidates"] += v["candidates"]; tot["training_pairs"] += v["training_pairs"]
        for s, n in v["replay_status"].items(): st[s] += n
        for s, n in v.get("pre_replay_rejections", {}).items(): pre[s] += n
        for k, d in v.get("replay_status_by_call_index", {}).items():
            for s, n in d.items(): k_st[k][s] += n
        for e, n in c["exit_status"].items(): tot[f"exit:{e}"] += n
    print(f"batches: {len(funnel['batches'])}   MASTER TRAINING PAIRS: {funnel.get('master_pairs', 0)}")
    print("funnel (cumulative):")
    print(f"  trajectories collected            {tot['trajectories']}   exits: " +
          ", ".join(f"{k[5:]} {v}" for k, v in tot.items() if k.startswith("exit:")))
    print(f"  solved (two-sided)                {tot['solved']}")
    print(f"  consultations answered            {tot['consultations_answered']}  (in solved trajectories: {tot['consultations_in_solved']})")
    print(f"  rejected before replay            {dict(pre)}")
    print(f"  candidates (passed A, strict, V)  {tot['candidates']}")
    print(f"  replay verdicts                   {dict(st)}")
    print(f"  training pairs                    {tot['training_pairs']}")
    if k_st:
        print("  by call_index:", {k: dict(v) for k, v in sorted(k_st.items())})
    # acceptance breakdowns over every verified candidate of every batch
    rows = []
    for b in funnel["batches"]:
        vf = Path(args.pairs_dir) / f"{b}_verified.jsonl"
        if vf.exists():
            rows += [json.loads(l) for l in vf.read_text().splitlines() if l.strip()]
    if rows:
        def _rate(group):
            n = len(group); a = sum(1 for r in group if r.get("status") in TRAIN_OK)
            return f"{a}/{n}"
        by = lambda key: collections.defaultdict(list)  # noqa: E731
        repo, step, length = by(0), by(0), by(0)
        for r in rows:
            repo[r["instance_id"].split(".")[0]].append(r)
            step[f"step<={int(r.get('step') or 0)//12*12+11}"].append(r)
            length[f"q- chars<={(len(r.get('rejected') or '')//300+1)*300}"].append(r)
        print("acceptance by repository:", {k: _rate(v) for k, v in sorted(repo.items())})
        print("acceptance by consultation step (difficulty proxy):", {k: _rate(v) for k, v in sorted(step.items())})
        print("acceptance by question length:", {k: _rate(v) for k, v in sorted(length.items())})
        flagged = sum(1 for r in rows if r.get("flag_altered_conditions"))
        print(f"candidates flagged for altered technical conditions (inspect): {flagged}/{len(rows)}")



def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("append"); a.add_argument("--batch", required=True); a.add_argument("--collect-dir", required=True)
    a.add_argument("--candidates", required=True); a.add_argument("--verified", required=True)
    a.add_argument("--pairs-dir", default="output/pairs")
    r = sub.add_parser("report"); r.add_argument("--pairs-dir", default="output/pairs")
    e = sub.add_parser("export"); e.add_argument("--pairs-dir", default="output/pairs")
    e.add_argument("--runs-root", default="output/runs"); e.add_argument("--out", required=True)
    e.add_argument("--solve-steps-out"); e.add_argument("--per-traj", type=int, default=6)
    e.add_argument("--char-budget", type=int, default=9000); e.add_argument("--kappa-p1", type=int, default=2)
    e.add_argument("--max-chars", type=int, default=2000); e.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    {"append": append, "report": report, "export": export}[args.cmd](args)


if __name__ == "__main__":
    main()
