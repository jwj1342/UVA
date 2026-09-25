#!/usr/bin/env python3
"""The DPO baseline's data (ablation on the training strategy).

Same accepted pairs as the mixed objective, but as preferences: prompt = the interaction history h up to the
consultation (read from the rollouts, or from the `prompt` field of exported rows), chosen = the
consultation in its deployment form with q+, rejected = the same with q-. DPO therefore trains on
the contrast between q+ and q-, whereas the mixed objective uses q- only to select pairs. Conversational
schema for TRL:
    {"prompt": [messages], "chosen": [assistant ask(q+)], "rejected": [assistant ask(q-)]}

    python -m uva.data.build_dpo_pairs --pairs output/pairs/master_pairs.jsonl \\
        --runs-root output/runs --out output/train/dpo_pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from uva.agent.ask_syntax import ask_action
from uva.data.elicitation import consultation_prompt, source_trajectory
from uva.data.pairs_master import TRAIN_OK


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--runs-root", default="output/runs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = all")
    ap.add_argument("--char-budget", type=int, default=9000)
    args = ap.parse_args()

    pairs = [json.loads(l) for l in Path(args.pairs).read_text().splitlines() if l.strip()]
    kept = n_no_prompt = n_skipped = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for p in pairs:
            if p.get("status") not in TRAIN_OK:
                n_skipped += 1
                continue
            qp, qm = (p.get("chosen") or "").strip(), (p.get("rejected") or "").strip()
            if not qp or not qm or qp == qm:
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
            fh.write(json.dumps({
                "prompt": prompt,
                "chosen": [{"role": "assistant", "content": ask_action(qp)}],
                "rejected": [{"role": "assistant", "content": ask_action(qm)}],
                "meta": {"instance_id": p["instance_id"], "call_index": p.get("call_index", 0),
                         "status": p.get("status")},
            }) + "\n")
            kept += 1
            if args.max_pairs and kept >= args.max_pairs:
                break
    print(f"[dpo] {len(pairs)} pairs in -> {kept} preference examples (skipped {n_skipped}, "
          f"no history {n_no_prompt}) -> {args.out}")
    if kept == 0:
        raise SystemExit("no DPO examples produced")


if __name__ == "__main__":
    main()
