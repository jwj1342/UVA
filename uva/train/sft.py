#!/usr/bin/env python3
"""LoRA supervised fine-tuning on the mixed supervision set.

Input: uva.data.build_sft_mix output, {"prompt": [messages], "chosen": <completion>, "kind",
"loss_span"?}. Each example is rendered with the student's chat template (thinking off) and the
loss covers only the loss span: the tokens of q+ for question examples, the whole action for
solve-step examples. Prompt, padding and scaffolding never carry loss. The two example types are
one dataset, so their weight is their example ratio.

Adapter: rank-16 LoRA (alpha 32, dropout 0.05) on all linear projections of a 4-bit nf4 base.
Budget: --max-steps optimizer updates (the recipes give every condition the same), else epochs.

    python -m uva.train.sft --model-path <base> --data-path output/train/sft_mix.jsonl \\
        --output-dir output/models/ours --max-steps 339 --learning-rate 7e-5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                          TrainingArguments)


def load(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb, device_map="auto",
                                             trust_remote_code=True)
    m = prepare_model_for_kbit_training(m)
    m.config.use_cache = False
    return m, tok


def render(tok, messages, add_gen: bool) -> str:
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_gen,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_gen)


def encode(tok, ex: dict, max_length: int) -> dict | None:
    """input_ids and labels (-100 outside the loss span, plus the end-of-turn token)."""
    chosen = ex["chosen"][0]["content"] if isinstance(ex["chosen"], list) else ex["chosen"]
    prompt = render(tok, ex["prompt"], True)
    full = render(tok, ex["prompt"] + [{"role": "assistant", "content": chosen}], False)
    span = ex.get("loss_span") or chosen
    start = full.find(span, len(prompt)) if full.startswith(prompt) else -1
    if start < 0:
        return None
    end = start + len(span)
    enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    labels = [tid if (a < end and b > start) else -100 for tid, (a, b) in zip(ids, offsets)]
    eot = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    for i in range(len(ids) - 1, -1, -1):
        if offsets[i][0] >= end and ids[i] in eot:
            labels[i] = ids[i]
            break
    ids, labels = ids[-max_length:], labels[-max_length:]  # the span sits at the end
    return None if all(l == -100 for l in labels) else {"input_ids": ids, "labels": labels}


def collate(batch, pad_id: int) -> dict:
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k], labels[i, :k], attn[i, :k] = torch.tensor(b["input_ids"]), torch.tensor(b["labels"]), 1
    return {"input_ids": ids, "labels": labels, "attention_mask": attn}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-epochs", type=float, default=3.0)
    p.add_argument("--max-steps", type=int, default=0, help="fixed update budget; 0 = use --num-epochs")
    p.add_argument("--learning-rate", type=float, default=7e-5)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--max-length", type=int, default=3072)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preview", type=int, default=8, help="greedy generations from question prompts after training")
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    model, tok = load(args.model_path)
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules="all-linear",
                                             lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM"))
    raw = [json.loads(l) for l in Path(args.data_path).read_text().splitlines() if l.strip()]
    recs = [r for r in (encode(tok, ex, args.max_length) for ex in raw) if r is not None]
    kinds = {}
    for ex in raw:
        kinds[ex.get("kind", "?")] = kinds.get(ex.get("kind", "?"), 0) + 1
    print(f"[sft] {len(recs)} examples of {len(raw)} {kinds}")

    targs = TrainingArguments(
        output_dir=str(out), num_train_epochs=args.num_epochs, max_steps=args.max_steps or -1,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, lr_scheduler_type="cosine", warmup_ratio=0.1,
        bf16=True, logging_steps=5, save_strategy="no", seed=args.seed, report_to="none",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=recs,
                      data_collator=lambda b: collate(b, tok.pad_token_id))
    trainer.train()
    final = out / "final_model"
    model.save_pretrained(str(final))
    tok.save_pretrained(str(final))
    (out / "train_config.json").write_text(json.dumps(vars(args), indent=2))

    if args.preview:
        model.config.use_cache = True
        model.eval()
        dev = next(model.parameters()).device
        for ex in [e for e in raw if e.get("kind") == "question"][:args.preview]:
            ids = tok(render(tok, ex["prompt"], True), return_tensors="pt", truncation=True,
                      max_length=args.max_length).input_ids.to(dev)
            with torch.no_grad():
                g = model.generate(ids, max_new_tokens=200, do_sample=False,
                                   pad_token_id=tok.pad_token_id or tok.eos_token_id)
            print("[sft preview]", repr(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)[:200]))
    print("[sft] ->", final)


if __name__ == "__main__":
    main()
