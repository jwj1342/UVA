#!/usr/bin/env python3
"""Put LoRA-merged text weights back into the base checkpoint's full layout, for serving.

The adapter only touches the text model's linear layers, so the servable checkpoint is the base's
weights with every text tensor replaced by its merged counterpart; the vision tower, config,
preprocessor and tokenizer come from the base unchanged. Every merged key must exist in the base
(same names, same shapes) and at least one tensor must actually be replaced, otherwise the script
refuses to write a checkpoint that would silently be the base.

    python -m uva.train.graft_text_weights --base <base> --merged output/models/ours-merged \\
        --out output/models/ours-served
"""
from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def _index(model_dir: str) -> dict[str, str]:
    idx = {}
    for f in sorted(glob.glob(f"{model_dir}/*.safetensors")):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                idx[k] = f
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base_idx, merged_idx = _index(args.base), _index(args.merged)
    extra = sorted(set(merged_idx) - set(base_idx))
    if extra:
        raise SystemExit(f"[graft] {len(extra)} merged keys absent from the base, e.g. {extra[:3]}; refusing")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    replaced = kept = 0
    for bf in sorted(glob.glob(f"{args.base}/*.safetensors")):
        tensors = {}
        with safe_open(bf, framework="pt") as f:
            for k in f.keys():
                if k in merged_idx:
                    with safe_open(merged_idx[k], framework="pt") as g:
                        t = g.get_tensor(k)
                    assert tuple(t.shape) == tuple(f.get_slice(k).get_shape()), k
                    tensors[k] = t.contiguous(); replaced += 1
                else:
                    tensors[k] = f.get_tensor(k).contiguous(); kept += 1
        save_file(tensors, str(out / Path(bf).name), metadata={"format": "pt"})
    if replaced == 0:
        raise SystemExit("[graft] no text tensor was replaced; the output would equal the base. Refusing.")
    for fn in Path(args.base).iterdir():
        if fn.is_file() and not fn.name.endswith(".safetensors"):
            shutil.copy2(fn, out / fn.name)  # config, tokenizer, preprocessor, shard index
    print(f"[graft] replaced {replaced} text tensors, kept {kept} base tensors -> {out}")


if __name__ == "__main__":
    main()
