#!/usr/bin/env python3
"""Merge a trained LoRA adapter into the base weights -> a standalone checkpoint directory.

    python -m uva.train.merge_lora --base <base> --adapter output/models/ours/final_model \\
        --out output/models/ours-merged

For a multimodal base (Qwen3.5 ships a vision tower next to the text model) the causal-LM merge
is text-only and its config names a text-only architecture that vLLM does not serve; run
uva.train.graft_text_weights afterwards to put the merged text weights back into the base's
full layout.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cpu",
                                                trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base, trust_remote_code=True).save_pretrained(str(out))
    print(f"[merge] -> {out}")


if __name__ == "__main__":
    main()
