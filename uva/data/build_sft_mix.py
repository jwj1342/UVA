#!/usr/bin/env python3
"""The supervision set for the mixed objective: question examples plus solve-step examples.

  question  prompt = the history h up to the consultation (the rollout's own system prompt, the task,
            the turns before it), chosen = `ask_expert "<q+>"`, loss on the tokens of q+ (`loss_span`)
  solve     prompt = a solved rollout's messages up to an ordinary action, chosen = that action;
            consultation turns are never targets

The mixing weight of the mixed objective is the example ratio --solve-per-question. Ablation switches:
--solve-per-question 0 (question-only), --no-replay-gate (every pre-replay candidate),
--max-pairs N (nested data budgets). Rows that carry a `prompt` (pairs_master export) need no
rollout; --solve-steps takes pre-extracted solve-step examples instead of --solved-runs.

    python -m uva.data.build_sft_mix --pairs output/pairs/master_pairs.jsonl \
        --solved-runs 'output/runs/*_collect' --out output/train/sft_mix.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

from uva.agent.ask_syntax import ask_action
from uva.data.elicitation import consultation_prompt, solve_step_examples, source_trajectory
from uva.data.pairs_master import TRAIN_OK


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True, help="master_pairs.jsonl (or a candidates file with --no-replay-gate)")
    ap.add_argument("--solved-runs", nargs="*", default=[], help="scored collection run dir(s) or globs")
    ap.add_argument("--solve-steps", help="pre-extracted solve-step examples (jsonl) instead of --solved-runs")
    ap.add_argument("--runs-root", default="output/runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--solve-per-question", type=float, default=2.0)
    ap.add_argument("--no-replay-gate", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = all")
    ap.add_argument("--char-budget", type=int, default=9000, help="history cap in characters; middle turns dropped")
    ap.add_argument("--per-traj", type=int, default=3, help="max solve-steps sampled per solved rollout")
    ap.add_argument("--scorer", choices=["smith", "verified"], default="smith")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    questions, n_skipped, n_no_prompt = [], 0, 0
    for line in Path(args.pairs).read_text().splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        if not args.no_replay_gate and p.get("status") not in TRAIN_OK:
            n_skipped += 1
            continue
        prompt = p.get("prompt")
        if prompt is None and p.get("run"):
            src = source_trajectory(args.runs_root, p["run"], p["instance_id"])
            if src.exists():
                prompt = consultation_prompt(src, args.char_budget, int(p.get("call_index", 0)))
        if prompt is None:
            n_no_prompt += 1
            continue
        q_plus = (p.get("chosen") or "").strip()
        completion = ask_action(q_plus)
        questions.append({"prompt": prompt, "chosen": completion, "kind": "question",
                          "loss_span": q_plus.replace('"', "'"),
                          "instance_id": p["instance_id"], "call_index": p.get("call_index", 0)})
        if args.max_pairs and len(questions) >= args.max_pairs:
            break

    if args.solve_steps:
        pool = [json.loads(l) for l in Path(args.solve_steps).read_text().splitlines() if l.strip()]
    else:
        run_dirs = sorted(d for pat in args.solved_runs for d in glob.glob(pat) if Path(d).is_dir())
        pool = solve_step_examples(run_dirs, args.char_budget, args.per_traj, rng, args.scorer)
    rng.shuffle(pool)
    n_solve = int(round(args.solve_per_question * len(questions)))
    solve = pool[:n_solve]

    mixed = questions + solve
    rng.shuffle(mixed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for r in mixed:
            fh.write(json.dumps(r) + "\n")
    print(f"[sft-mix] questions={len(questions)} (skipped by replay gate {n_skipped}, no history {n_no_prompt}), "
          f"solve-steps available={len(pool)} used={len(solve)}, total={len(mixed)} -> {args.out}")
    if n_solve > len(pool):
        print(f"[sft-mix] WARNING: only {len(pool)} solve-steps for a requested {n_solve}; raise --per-traj "
              f"or add solved runs")


if __name__ == "__main__":
    main()
