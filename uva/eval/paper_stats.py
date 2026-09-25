#!/usr/bin/env python3
"""Solve statistics from scored runs: pass@k, solve@k, exact McNemar and a paired bootstrap interval.

Each comma-separated run of an arm is one sample; every instance of the pool counts and an instance
without a scored patch is a failure. pass@k is the unbiased estimator from n >= k samples; the paired
contrast is on solve@k (any of the first k samples), with discordant counts (b, c), the exact McNemar
p-value and a bootstrap interval over instances.

    python -m uva.eval.paper_stats passk --pool data/pools/verified_500.jsonl \
        --arm solo=RUN_k0,RUN_k1,RUN_k2 --arm ours=RUN_k0,RUN_k1,RUN_k2 --k 1 3
    python -m uva.eval.paper_stats mcnemar --pool ... --arm ours=... --arm natural=... --k 3
"""
from __future__ import annotations

import argparse
import glob
import json
from math import comb
from pathlib import Path

from uva.privacy import leakage as L


def load_pool(path: str) -> list[str]:
    return [json.loads(l)["instance_id"] if l.strip().startswith("{") else l.strip()
            for l in Path(path).read_text().splitlines() if l.strip()]


def arm_samples(spec: str, pool: list[str], scorer: str) -> dict[str, list[int]]:
    """{instance: [outcome per sample]}; each comma-separated entry (dir or glob) is one sample."""
    out = {i: [] for i in pool}
    for k, part in enumerate(spec.split(",")):
        dirs = sorted(glob.glob(part)) or [part]
        res: dict[str, dict] = {}
        for d in dirs:
            ev = Path(d) / "eval_results.json"
            if ev.exists():
                res.update({i: v for i, v in json.loads(ev.read_text()).items() if isinstance(v, dict)})
        for i in pool:
            v = res.get(i)
            ok = bool(v and v.get("resolved") and (v.get("reward", 1.0) or 0) >= 0.999) if scorer == "tb" \
                else bool(v and v.get("resolved") and v.get("patch_applied") and not v.get("harness_error"))
            out[i].append(int(ok))
    return out


def pass_at_k(samples: list[int], k: int) -> float | None:
    n, c = len(samples), sum(samples)
    if n < k:
        return None
    return 1.0 - (comb(n - c, k) / comb(n, k) if n - c >= k else 0.0)


def solve_at_k(samples: list[int], k: int) -> int | None:
    return None if len(samples) < k else int(any(samples[:k]))


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(m + 1)) * 0.5 ** n)


def cmd_passk(arms: dict, ks: list[int]) -> None:
    for name, d in arms.items():
        ns = sorted({len(v) for v in d.values()})
        line = [f"{name}: instances={len(d)} samples/instance={ns}"]
        for k in ks:
            vals = [pass_at_k(v, k) for v in d.values()]
            ok = [v for v in vals if v is not None]
            if ok:
                line.append(f"pass@{k}={100 * sum(ok) / len(ok):.1f}")
            sk = [s for s in (solve_at_k(v, k) for v in d.values()) if s is not None]
            if sk and k > 1:
                line.append(f"solve@{k}={sum(sk)}/{len(sk)}")
        per_sample = [sum(v[j] for v in d.values() if len(v) > j) for j in range(max(ns) if ns else 0)]
        line.append(f"per-sample resolved={per_sample}")
        print("  ".join(line))


def cmd_mcnemar(arms: dict, a: str, b_: str, k: int, boot: int = 2000, seed: int = 0) -> None:
    import random
    A, B = arms[a], arms[b_]
    diffs = []
    b = c = both = neither = skipped = 0
    for i in A:
        sa, sb = solve_at_k(A[i], k), solve_at_k(B[i], k)
        if sa is None or sb is None:
            skipped += 1
            continue
        diffs.append(sa - sb)
        if sa and not sb: b += 1
        elif sb and not sa: c += 1
        elif sa and sb: both += 1
        else: neither += 1
    n = b + c + both + neither
    print(f"{a} vs {b_} on solve@{k}: paired n={n} ({skipped} instances with <{k} samples)")
    print(f"  delta = {(b - c) / n:+.3f}  discordant (b={b}, c={c})  both={both} neither={neither}")
    print(f"  exact McNemar two-sided p = {mcnemar_exact(b, c):.4f}   d=(b+c)/n = {(b + c) / n:.3f}")
    if boot and n:
        rng = random.Random(seed)
        means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(boot))
        lo, hi = means[int(0.025 * boot)], means[int(0.975 * boot) - 1]
        print(f"  paired bootstrap 95% interval for delta: [{lo:+.3f}, {hi:+.3f}]  ({boot} resamples over instances)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("passk", "mcnemar"):
        s = sub.add_parser(name)
        s.add_argument("--pool", required=True, help="assigned instances (jsonl or id list)")
        s.add_argument("--arm", action="append", required=True, help="NAME=dir_or_glob[,dir_or_glob...] one per sample")
        s.add_argument("--scorer", choices=["verified", "smith", "tb"], default="verified")
        s.add_argument("--k", type=int, nargs="+", default=[1, 3])
        s.add_argument("--bootstrap", type=int, default=2000, help="paired bootstrap resamples (mcnemar)")
    args = ap.parse_args()
    pool = load_pool(args.pool)
    arms = {}
    for spec in args.arm:
        name, dirs = spec.split("=", 1)
        arms[name] = arm_samples(dirs, pool, args.scorer)
    if args.cmd == "passk":
        cmd_passk(arms, args.k)
    else:
        names = list(arms)
        if len(names) != 2:
            raise SystemExit("mcnemar needs exactly two --arm")
        cmd_mcnemar(arms, names[0], names[1], args.k[-1] if args.k else 3, args.bootstrap)


if __name__ == "__main__":
    main()
