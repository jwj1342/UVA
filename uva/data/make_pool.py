#!/usr/bin/env python3
"""Rebuild the task pools from the released instance-id lists.

The harness reads local JSON Lines files with the fields
    instance_id, image_name, problem_statement, FAIL_TO_PASS, PASS_TO_PASS   (+ repo, base_commit)
This pulls those records from the public Hugging Face datasets and writes them in the order of
the id list, so slices (`--slice a:b`) reproduce the batches used in the paper.

    python -m uva.data.make_pool swesmith  data/swesmith_train_ids.txt   data/pools/swesmith_train.jsonl
    python -m uva.data.make_pool swesmith  data/swesmith_heldout_ids.txt data/pools/swesmith_heldout.jsonl
    python -m uva.data.make_pool verified  data/verified_500_order.txt   data/pools/verified_500.jsonl

SWE-smith's per-instance bug lives on a git branch named after the instance inside its repository
image (`image_name`); the collection config checks it out before the agent starts. SWE-bench
Verified images are already at the instance's base commit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASETS = {"swesmith": ("SWE-bench/SWE-smith", "train"),
            "verified": ("princeton-nlp/SWE-Bench_Verified", "test")}
FIELDS = ("instance_id", "repo", "image_name", "problem_statement", "FAIL_TO_PASS", "PASS_TO_PASS", "base_commit")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", choices=list(DATASETS))
    ap.add_argument("ids")
    ap.add_argument("out")
    args = ap.parse_args()
    from datasets import load_dataset

    name, split = DATASETS[args.pool]
    ids = [l.strip() for l in Path(args.ids).read_text().splitlines() if l.strip()]
    want = set(ids)
    rows = {}
    for r in load_dataset(name, split=split):
        if r["instance_id"] in want:
            rec = {k: r[k] for k in FIELDS if k in r}
            for k in ("FAIL_TO_PASS", "PASS_TO_PASS"):
                if isinstance(rec.get(k), str):
                    rec[k] = json.loads(rec[k])
            rows[r["instance_id"]] = rec
    missing = [i for i in ids if i not in rows]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for i in ids:
            if i in rows:
                fh.write(json.dumps(rows[i]) + "\n")
    print(f"wrote {len(ids) - len(missing)}/{len(ids)} records -> {args.out}"
          + (f"; missing {missing[:5]}..." if missing else ""))


if __name__ == "__main__":
    main()
