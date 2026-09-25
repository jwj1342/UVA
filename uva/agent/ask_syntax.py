"""Syntax of the `ask_expert` command, shared by the agent and by the offline readers.

Kept free of harness imports so that data construction and measurement can locate consultations
in a recorded trajectory without loading the agent stack.
"""
from __future__ import annotations

import re

# `ask_expert <question>`, quoted or bare. The ask may also arrive as the tail of a compound
# command (`cd /testbed && ask_expert "..."`); the prefix is dropped rather than executed, since
# it cannot change the question and running it could change state the question no longer
# reflects.
ASK = re.compile(
    r"^\s*(?:(?P<prefix>.*?)(?:&&|\|\||;|\||\n)\s*)?ask_expert\b\s*(?P<q>.*)$", re.DOTALL)


def parse_ask(command: str) -> str | None:
    """The question of an ask command, or None if the command is not an ask.

    Note: the harness's action parser closes a command block at the FIRST inner fence, so a
    question that contains a code block arrives here truncated; `looks_truncated` flags it."""
    m = ASK.match(command or "")
    if not m:
        return None
    q = m.group("q").strip()
    if len(q) >= 2 and q[0] == q[-1] and q[0] in "\"'":
        q = q[1:-1]
    return q


def looks_truncated(q: str) -> bool:
    t = (q or "").rstrip()
    return t.endswith((":", "```", "`")) or t.count("```") % 2 == 1


def is_ask_command(command: str) -> bool:
    return parse_ask(command) is not None


def ask_action(question: str) -> str:
    """The consultation as the student emits it at deployment: one command block whose command is
    `ask_expert "<question>"`. This is the completion form of a question example in training, so
    the trained behaviour is exactly the deployed one."""
    q = (question or "").strip().replace('"', "'")
    return f'```mswea_bash_command\nask_expert "{q}"\n```'
