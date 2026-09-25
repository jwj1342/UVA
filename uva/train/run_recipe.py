#!/usr/bin/env python3
"""One training condition from a recipe in configs/train/: build the data, train, merge, graft.

    python -m uva.train.run_recipe configs/train/ours.yaml --base <base> --solved-runs 'output/runs/*_collect'

`train.update_budget` is the number of optimizer updates shared by every condition; without it
`train.epochs` applies. --from {train,merge} skips earlier steps.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def run(cmd: list[str]) -> None:
    print("+", " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recipe")
    ap.add_argument("--base", required=True, help="base student checkpoint")
    ap.add_argument("--solved-runs", nargs="*", default=[], help="scored collection runs (solve-step source)")
    ap.add_argument("--runs-root", default="output/runs")
    ap.add_argument("--out-root", default="output/models")
    ap.add_argument("--from", dest="start", choices=["build", "train", "merge"], default="build")
    args = ap.parse_args()
    r = yaml.safe_load(Path(args.recipe).read_text())
    name = r["name"]
    out = Path(args.out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    py = [sys.executable, "-m"]
    data = r["data"]; tr = r["train"]

    if r["objective"] == "sft":
        data_path = out / "sft_mix.jsonl"
        if args.start == "build":
            cmd = py + ["uva.data.build_sft_mix", "--pairs", data["pairs"], "--runs-root", args.runs_root,
                        "--out", str(data_path), "--solve-per-question", str(data["solve_per_question"]),
                        "--char-budget", str(data.get("char_budget", 9000)), "--seed", str(r.get("seed", 42))]
            if args.solved_runs:
                cmd += ["--solved-runs", *args.solved_runs]
            elif data.get("solve_steps"):
                cmd += ["--solve-steps", data["solve_steps"]]
            else:
                cmd += ["--solved-runs", *data.get("solved_runs", ["output/runs/*_collect"])]
            if not data.get("replay_gate", True):
                cmd.append("--no-replay-gate")
            if data.get("max_pairs"):
                cmd += ["--max-pairs", str(data["max_pairs"])]
            run(cmd)
        if args.start in ("build", "train"):
            run(py + ["uva.train.sft", "--model-path", args.base, "--data-path", str(data_path),
                      "--output-dir", str(out), "--num-epochs", str(tr.get("epochs", 3)),
                      "--max-steps", str(tr.get("update_budget", 0)),
                      "--learning-rate", str(tr["learning_rate"]), "--max-length", str(tr.get("max_length", 3072)),
                      "--gradient-accumulation-steps", str(tr.get("gradient_accumulation_steps", 8)),
                      "--lora-r", str(tr["lora"]["r"]), "--lora-alpha", str(tr["lora"]["alpha"]),
                      "--lora-dropout", str(tr["lora"]["dropout"]), "--seed", str(r.get("seed", 42))])
    elif r["objective"] == "dpo":
        data_path = out / "dpo_pairs.jsonl"
        if args.start == "build":
            cmd = py + ["uva.data.build_dpo_pairs", "--pairs", data["pairs"], "--runs-root", args.runs_root,
                        "--out", str(data_path), "--char-budget", str(data.get("char_budget", 9000))]
            if data.get("max_pairs"):
                cmd += ["--max-pairs", str(data["max_pairs"])]
            run(cmd)
        if args.start in ("build", "train"):
            run(py + ["uva.train.dpo", "--model-path", args.base, "--data-path", str(data_path),
                      "--output-dir", str(out), "--num-epochs", str(tr.get("epochs", 3)),
                      "--max-steps", str(tr.get("update_budget", 0)),
                      "--learning-rate", str(tr["learning_rate"]), "--beta", str(tr["beta"]),
                      "--max-length", str(tr.get("max_length", 3072)),
                      "--max-prompt-length", str(tr.get("max_prompt_length", 2816)),
                      "--lora-r", str(tr["lora"]["r"]), "--lora-alpha", str(tr["lora"]["alpha"]),
                      "--lora-dropout", str(tr["lora"]["dropout"]), "--seed", str(r.get("seed", 42))])
    else:
        raise SystemExit(f"unknown objective {r['objective']!r}")

    run(py + ["uva.train.merge_lora", "--base", args.base, "--adapter", str(out / "final_model"),
              "--out", str(out / "merged")])
    run(py + ["uva.train.graft_text_weights", "--base", args.base, "--merged", str(out / "merged"),
              "--out", str(out / "served")])
    print(f"[recipe] {name}: servable checkpoint -> {out / 'served'}")


if __name__ == "__main__":
    main()
