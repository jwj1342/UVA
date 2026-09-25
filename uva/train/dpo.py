#!/usr/bin/env python3
"""The DPO baseline : prefer q+ over q- on the same accepted pairs.

Data: uva.data.build_dpo_pairs output. The loss is on the completion tokens (the consultation
action with q+ versus q-), so it is question-level by construction; the identical scaffolding
around the two questions cancels in the log-ratio. Reference policy = the same 4-bit base with the
LoRA adapter disabled (frozen-reference likelihoods without loading a second model). Same adapter
shape and update budget (--max-steps) as uva.train.sft; the token-compute-matched variant scales
the budget down by the extra rejected and reference passes.

    python -m uva.train.dpo --model-path <base> --data-path output/train/dpo_pairs.jsonl \\
        --output-dir output/models/dpo
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


def build(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb, device_map="auto",
                                                 trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    return model, tokenizer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-epochs", type=float, default=3.0)
    p.add_argument("--max-steps", type=int, default=0, help="fixed update budget; 0 = use --num-epochs")
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=3072)
    p.add_argument("--max-prompt-length", type=int, default=2816)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    model, tokenizer = build(args.model_path)
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules="all-linear",
                      lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM")
    ds = load_dataset("json", data_files=args.data_path, split="train")
    print(f"[dpo] {len(ds)} preference examples")

    def _apply(messages, add_gen):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=add_gen, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_gen)

    def render(ex):
        prompt = _apply(ex["prompt"], True)
        out = {"prompt": prompt}
        for key in ("chosen", "rejected"):
            full = _apply(ex["prompt"] + ex[key], False)
            out[key] = full[len(prompt):] if full.startswith(prompt) else ex[key][0]["content"]
        return out

    ds = ds.map(render, remove_columns=[c for c in ds.column_names if c not in ("prompt", "chosen", "rejected")])
    ex0 = ds[0]
    assert ex0["prompt"].count("<think>") == ex0["prompt"].count("</think>"), "unclosed <think> in prompt"

    wanted = dict(
        output_dir=str(out), num_train_epochs=args.num_epochs, max_steps=args.max_steps or -1,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, lr_scheduler_type="cosine", warmup_ratio=0.1, beta=args.beta,
        max_length=args.max_length, max_prompt_length=args.max_prompt_length, truncation_mode="keep_end",
        bf16=True, logging_steps=5, save_strategy="no", seed=args.seed, report_to="none",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    valid = {f.name for f in dataclasses.fields(DPOConfig)}
    cfg = DPOConfig(**{k: v for k, v in wanted.items() if k in valid})
    trainer = DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=ds,
                         processing_class=tokenizer, peft_config=lora)
    print("[dpo] training...")
    trainer.train()
    final = out / "final_model"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    (out / "train_config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"[dpo] done -> {final}")


if __name__ == "__main__":
    main()
