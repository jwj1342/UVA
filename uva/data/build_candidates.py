#!/usr/bin/env python3
"""Candidates: rewrite each probed consultation of a solved rollout and apply the pre-replay checks.

For each probed consultation (q-, with reference r_h = the observations before it and inventory
I(r_h)), the teacher produces q+ = R(h, q-) (`--target abstract`) or the pseudonymizer T(q-)
(`--target pseudonym`, Host-SFT). A candidate survives only if
    A(q+)     no hard secret, at most --kappa-p1 detector-flagged P1 spans
    strict    L_P1(q+; r_h) < L_P1(q-; r_h) on the provenance count
    V         nonempty, bounded, plain prose (no fence, no shell command), no IDENT_n placeholder,
              a closing sentence that poses the decision
then byte-identical histories are deduplicated and at most --max-per-task candidates kept per task.
Rejections go to <out>.rejected.jsonl with their stage. Each candidate records both questions'
counts, the consultation's position and sampling seed, and a flag when the rewrite drops a technical
condition of the original (for inspection, not rejection).

    python -m uva.data.build_candidates --runs output/runs/b0_collect --target abstract \
        --rewriter-model <teacher> --out output/pairs/b0_candidates.jsonl
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

from uva.data import teacher
from uva.data.elicitation import history_text
from uva.privacy.leakage import consultations, reference_context
from uva.privacy.detector import Pseudonymizer
from uva.privacy import leakage as L

_PLACEHOLDER = re.compile(r"\b(?:SECRET|URL|PATH|MODULE|IDENT|FUNC|VAR|CLASS)_\d+\b")
_SHELL_START = re.compile(
    r"^\s*(cd|ls|cat|grep|rg|python3?|pytest|pip|git|sed|awk|echo|printf|rm|mv|cp|find|"
    r"chmod|chown|sudo|bash|sh|make|export|touch|mkdir|head|tail|diff|apply|patch)\b")
# the closing sentence must pose the decision: a question, or a request for a judgement
# technical conditions a rewrite must not lose: literals, error names, comparison operators
_CONDITION = re.compile(r"\b\d+(?:\.\d+)?\b|\b\w+(?:Error|Exception|Warning)\b|==|!=|<=|>=|\bNone\b|\bTrue\b|\bFalse\b")
_REQUEST = re.compile(
    r"\b(should|whether|which|what|how|why|is it|are there|could|would|can you|confirm|review|"
    r"advise|suggest|recommend|please|verify|correct|approach|need|expected|appropriate|"
    r"safe|acceptable|prefer|better|right)\b", re.I)


def validity(q_plus: str, *, max_chars: int) -> str | None:
    """V(h, q+, q-): None if valid, else the failing clause."""
    t = (q_plus or "").strip()
    if not t:
        return "empty"
    if len(t) > max_chars:
        return "too_long"
    if "```" in t:
        return "code_fence"
    if _SHELL_START.match(t):
        return "shell_command"
    if _PLACEHOLDER.search(t):
        return "opaque_placeholder"
    parts = re.split(r"(?<=[.?!])\s+", t)
    last = parts[-1] if parts else t
    if not (last.endswith("?") or _REQUEST.search(last)):
        return "no_closing_decision"
    return None


def altered_conditions(q_minus: str, q_plus: str) -> bool:
    """True if q- states technical conditions and q+ keeps none of them (flag for inspection)."""
    before = set(_CONDITION.findall(q_minus or ""))
    return bool(before) and not (before & set(_CONDITION.findall(q_plus or "")))


def _resolved_ids(run_dir: Path, scorer: str) -> tuple[set[str], bool]:
    ev = run_dir / "eval_results.json"
    if not ev.exists():
        return set(), False
    data = json.loads(ev.read_text())
    return ({k for k, v in data.items() if L.strict_resolved(v, scorer)}, True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="collection run dir(s) or globs, each scored")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", choices=["abstract", "pseudonym"], default="abstract",
                    help="abstract: teacher rewrite (ours); pseudonym: host transform T (Host-SFT ablation)")
    ap.add_argument("--rewriter-model", default="",
                    help="teacher model name at TEACHER_API_BASE (required for --target abstract)")
    ap.add_argument("--kappa-p1", type=int, default=2,
                    help="A(q+): at most this many detector-only P1 spans may remain")
    ap.add_argument("--max-chars", type=int, default=2000, help="V: length bound on q+")
    ap.add_argument("--max-per-task", type=int, default=5, help="cap on candidates per task (instance)")
    ap.add_argument("--char-budget", type=int, default=9000, help="history shown to the teacher (characters)")
    ap.add_argument("--scorer", choices=["smith", "verified"], default="smith")
    args = ap.parse_args()
    if args.target == "abstract" and not args.rewriter_model:
        ap.error("--rewriter-model is required for --target abstract (the teacher R is a separate "
                 "model from the expert E; do not default it to the expert)")
    base = key = ""
    if args.target == "abstract":
        base, key = teacher.endpoint()

    run_dirs = sorted(d for pat in args.runs for d in glob.glob(pat) if Path(d).is_dir())
    cands: dict[str, dict] = {}
    rejected: list[dict] = []
    seen_prefix: set[str] = set()
    per_task: dict[str, int] = {}
    n_traj = n_solved = n_probed = 0
    for run_dir in run_dirs:
        rd = Path(run_dir)
        resolved, have_eval = _resolved_ids(rd, args.scorer)
        if not have_eval:
            print(f"[candidates] skip {rd.name}: no eval_results.json (score the run first)", file=sys.stderr)
            continue
        for traj in sorted(rd.glob("*/*.traj.json")):
            n_traj += 1
            d = json.loads(traj.read_text())
            iid = d.get("instance_id") or traj.parent.name
            if iid not in resolved:
                continue  # only consultations from successful rollouts are candidates
            n_solved += 1
            thash = hashlib.sha256(traj.read_bytes()).hexdigest()[:16]
            msgs = d.get("messages", [])
            seed = (((d.get("info") or {}).get("config") or {}).get("model") or {}).get("model_kwargs", {}).get("seed")
            cons = consultations(msgs)
            calls = ((d.get("info") or {}).get("ask_expert") or {}).get("calls", [])
            for idx, c in enumerate(calls):
                if not (c.get("forced") and L.is_clean_forced(c)) or idx >= len(cons):
                    continue
                cut, kind = cons[idx]
                if kind != "forced":
                    continue
                n_probed += 1
                q_minus = c["question"].strip()
                ref = reference_context(msgs, cut)
                prefix_key = hashlib.sha256(json.dumps(
                    [(m.get("role"), m.get("content")) for m in msgs[:cut]]).encode()).hexdigest()
                inv = L.reference_inventory(ref)
                lp1_minus, _ = L.leaked_entities_ref(q_minus, inv)
                oco_minus = L.oco_ref(q_minus, ref)[0]
                rec = {"instance_id": iid, "run": rd.name, "traj_sha16": thash, "call_index": idx,
                       "step": c.get("step"), "seed": seed, "target": args.target, "rejected": q_minus,
                       "lp1_rejected": lp1_minus, "oco_rejected": oco_minus,
                       "crit_rejected": L.crit(q_minus),
                       "detector_only_rejected": L.leaked_entities(q_minus)[0],
                       "inventory_size": len(inv)}

                def reject(stage: str, q_plus: str = "", **extra) -> None:
                    rejected.append({**rec, "chosen": q_plus, "reject_stage": stage, **extra})

                if args.target == "abstract":
                    try:
                        q_plus = teacher.rewrite(q_minus, history_text(msgs, cut, args.char_budget),
                                                 args.rewriter_model, base, key)
                    except Exception as e:  # noqa: BLE001 -- one failed call must not stop the batch
                        reject("rewrite_failed", error=f"{type(e).__name__}: {e}")
                        continue
                else:
                    q_plus = Pseudonymizer().transform(q_minus, include_p1=True)
                q_plus = q_plus.strip()
                if q_plus == q_minus:
                    reject("identical", q_plus)
                    continue
                # A(q+)
                crit_plus = L.crit(q_plus)
                det_plus = L.leaked_entities(q_plus)[0]
                if crit_plus > 0 or det_plus > args.kappa_p1:
                    reject("privacy_predicate", q_plus, crit_chosen=crit_plus, detector_only_chosen=det_plus)
                    continue
                # strict reduction, same reference
                lp1_plus, _ = L.leaked_entities_ref(q_plus, inv)
                if not lp1_plus < lp1_minus:
                    reject("no_strict_reduction", q_plus, lp1_chosen=lp1_plus)
                    continue
                # V (the pseudonym target legitimately carries IDENT_n)
                v = validity(q_plus, max_chars=args.max_chars)
                if v == "opaque_placeholder" and args.target == "pseudonym":
                    v = None
                if v is not None:
                    reject(f"validity:{v}", q_plus, lp1_chosen=lp1_plus)
                    continue
                if prefix_key in seen_prefix:
                    reject("duplicate_prefix", q_plus, lp1_chosen=lp1_plus)
                    continue
                if per_task.get(iid, 0) >= args.max_per_task:
                    reject("per_task_cap", q_plus, lp1_chosen=lp1_plus)
                    continue
                seen_prefix.add(prefix_key)
                per_task[iid] = per_task.get(iid, 0) + 1
                key_ = f"{iid}:{thash}:{idx}"
                cands[key_] = {**rec, "chosen": q_plus, "lp1_chosen": lp1_plus,
                               "oco_chosen": L.oco_ref(q_plus, ref)[0], "crit_chosen": crit_plus,
                               "detector_only_chosen": det_plus,
                               "chosen_chars": len(q_plus), "rejected_chars": len(q_minus),
                               "prefix_sha16": prefix_key[:16],
                               "flag_altered_conditions": altered_conditions(q_minus, q_plus),
                               "status": "candidate", "replay_verified": None}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for p in cands.values():
            fh.write(json.dumps(p) + "\n")
    rej_path = args.out.with_suffix(".rejected.jsonl")
    with rej_path.open("w") as fh:
        for p in rejected:
            fh.write(json.dumps(p) + "\n")
    from collections import Counter
    stages = Counter(r["reject_stage"] for r in rejected)
    print(f"[candidates] rollouts {n_traj}, solved {n_solved}, probed consultations {n_probed}, "
          f"candidates {len(cands)}, rejected {len(rejected)} {dict(stages)}", file=sys.stderr)
    print(f"[candidates] -> {args.out}  (rejections -> {rej_path})", file=sys.stderr)
    if not cands:
        raise SystemExit("no candidates")


if __name__ == "__main__":
    main()
