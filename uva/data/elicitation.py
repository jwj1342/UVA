"""Histories and examples from recorded rollouts.

`consultations` (uva.privacy.leakage) locates the k-th consultation in a rollout's messages;
`history_prompt` is the history h up to it, middle-truncated to a character budget that keeps the
system prompt, the task and the most recent turns; `history_text` renders it for the teacher;
`consultation_prompt` reads h for a probed consultation of a rollout file; `solve_step_examples`
samples (messages up to an action -> that action) from solved rollouts, never a consultation turn.
"""
from __future__ import annotations

import glob
import json
import random
from pathlib import Path

from uva.agent.ask_syntax import is_ask_command
from uva.privacy import leakage as L
from uva.privacy.leakage import consultations, reference_context  # noqa: F401  (re-exported)


def _char_total(msgs) -> int:
    return sum(len(m.get("content") or "") for m in msgs)


def truncate_middle(msgs: list[dict], budget: int, keep_head: int = 2) -> list[dict]:
    """Keep the first `keep_head` messages (system prompt and task) and as many recent turns as
    fit in `budget` characters; drop from the middle."""
    if _char_total(msgs) <= budget or len(msgs) <= keep_head + 1:
        return msgs
    head, tail = msgs[:keep_head], msgs[keep_head:]
    while _char_total(head + tail) > budget and len(tail) > 1:
        tail = tail[1:]
    return head + tail


def source_trajectory(runs_root, run: str, instance_id: str) -> Path:
    return Path(runs_root) / run / instance_id / f"{instance_id}.traj.json"


def _clean(msgs: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": m.get("content", "")} for m in msgs
            if m.get("role") in ("system", "user", "assistant") and isinstance(m.get("content"), str)]


def history_prompt(msgs: list[dict], cut: int, char_budget: int) -> list[dict]:
    """h for the consultation at message index `cut`, as a message list."""
    return truncate_middle(_clean(msgs[:cut]), char_budget)


def history_text(msgs: list[dict], cut: int, char_budget: int) -> str:
    """h rendered for the teacher: role-tagged turns, without the student's system prompt."""
    turns = [m for m in history_prompt(msgs, cut, char_budget) if m["role"] != "system"]
    return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in turns)


def consultation_prompt(src_traj, char_budget: int, call_index: int = 0) -> list[dict] | None:
    """h of the call_index-th consultation of a recorded rollout, which must be a probed one."""
    d = json.loads(Path(src_traj).read_text())
    msgs = d.get("messages", [])
    cons = consultations(msgs)
    if len(cons) <= call_index or cons[call_index][1] != "forced":
        return None
    return history_prompt(msgs, cons[call_index][0], char_budget)


def solve_step_examples(run_dirs, budget: int, per_traj: int, rng: random.Random,
                        scorer: str = "smith") -> list[dict]:
    """(prompt, action) examples from SOLVED rollouts, `per_traj` random action steps each."""
    out = []
    for run in run_dirs:
        ev = Path(run) / "eval_results.json"
        res = json.loads(ev.read_text()) if ev.exists() else {}
        for t in sorted(glob.glob(f"{run}/*/*.traj.json")):
            iid = Path(t).parent.name
            if not L.strict_resolved(res.get(iid, {}), scorer):
                continue
            msgs = json.loads(Path(t).read_text()).get("messages", [])
            idxs = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"
                    and (m.get("extra") or {}).get("actions")
                    and not any(is_ask_command(a.get("command", ""))
                                for a in (m.get("extra") or {}).get("actions", []))]
            rng.shuffle(idxs)
            for i in idxs[:per_traj]:
                prompt = history_prompt(msgs, i, budget)
                chosen = msgs[i].get("content") or ""
                if prompt and chosen.strip():
                    out.append({"prompt": prompt, "chosen": chosen, "kind": "solve", "instance_id": iid})
    return out
