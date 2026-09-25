"""The offline teacher R: q+ = R(h, q-). A separate model from the expert, used only in data
construction, called through an OpenAI-compatible endpoint (TEACHER_API_BASE, TEACHER_API_KEY) with
REWRITER_SYSTEM and one user message carrying the recent history and the request. The Privacy prompt
baseline puts REWRITER_SYSTEM verbatim into the student's system prompt.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

REWRITER_SYSTEM = """You rewrite a software engineer's help request so it can be sent to an outside \
expert WITHOUT leaking any private code detail, while staying just as useful to answer.

RULES:
- Remove EVERY private identifier: file paths, module/package names, class names, function/method \
names, variable/parameter names, and any verbatim code or literal values.
- Do NOT replace them with opaque placeholders (no IDENT_1, FUNC_2, etc.). Instead describe each by \
its ROLE and TYPE, e.g. "the constructor of a shared auth-token object", "a token-generator callable", \
"a boolean formatting option", "a context-manager's await hook".
- KEEP all engineering substance: the observed symptom, what was expected, the STRUCTURAL pattern of \
the bug (e.g. "two adjacent same-type constructor arguments are passed in swapped order"), what the \
engineer already ruled out, and the exact decision they need judged.
- The result MUST be answerable by an expert who cannot see the code: convey enough structure to \
reason about, without naming anything private.
- Keep it concise (2-5 sentences). Output ONLY the rewritten question, no preamble."""

USER_TEMPLATE = """The engineer's recent session (for your understanding only; do not quote it):

{history}

Rewrite this help request:

{question}"""


def endpoint() -> tuple[str, str]:
    base = os.environ.get("TEACHER_API_BASE", "").rstrip("/")
    key = os.environ.get("TEACHER_API_KEY", "")
    key_file = os.environ.get("TEACHER_API_KEY_FILE", "")
    if not key and key_file and Path(key_file).exists():
        key = Path(key_file).read_text().strip().splitlines()[-1].strip()
    if not base or not key:
        raise SystemExit("set TEACHER_API_BASE and TEACHER_API_KEY (or TEACHER_API_KEY_FILE)")
    return base, key


def rewrite(question: str, history: str, model: str, base: str, key: str, *, temperature: float = 0.3,
            max_tokens: int = 600, timeout: int = 120) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": REWRITER_SYSTEM},
                     {"role": "user", "content": USER_TEMPLATE.format(history=history or "(none)",
                                                                       question=question)}],
        "temperature": temperature, "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"].strip()
