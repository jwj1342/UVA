"""Detector rules and the deterministic pseudonymizer.

Recognizers for hard secrets (credential key/values, URLs, absolute paths, high-entropy strings) and
for identifiers (dotted module paths, identifier-shaped tokens that are not ordinary English words),
the pseudonymizer T used by the Host sanitizer policy (SECRET_n / URL_n / PATH_n / MODULE_n /
IDENT_n with a reverse map), and the source-lexeme n-grams behind OCO. Stdlib only.
"""

from __future__ import annotations

import builtins
import keyword
import re
import sys
from pathlib import Path

_RX_KEYVAL = re.compile(
    r"""(?ix)\b(?:api[_-]?key|token|secret|password|passwd|credential|provenance)\b\s*[:=]\s*['"]?[^\s'"]+"""
)

_RX_URL = re.compile(r"(?:https?|file)://[^\s'\"]+")

_PATH_SEGMENT = r"[^/\s'\":<>|]+"

_KNOWN_ABSOLUTE_ROOT = r"(?:data|etc|home|mnt|nfs|opt|private|project|root|scratch|srv|testbed|tmp|usr|var|workspace)"

_RX_POSIX_PATH = re.compile(
    rf"(?<![A-Za-z0-9_:/])/(?:{_KNOWN_ABSOLUTE_ROOT}(?:/{_PATH_SEGMENT})*|"
    rf"{_PATH_SEGMENT}/{_PATH_SEGMENT}(?:/{_PATH_SEGMENT})*)"
)

_RX_WINDOWS_PATH = re.compile(r"\b[A-Za-z]:\\(?:[^\s'\"\\]+\\)*[^\s'\"\\]+")

_RX_HIENT = re.compile(r"[A-Za-z0-9_+/=-]{16,}")

_RX_DOTTED = re.compile(r"\b[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*){2,}\b")

_RX_IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{5,}\b")

_RX_PLACEHOLDER = re.compile(
    r"(?:SECRET|URL|PATH|MODULE|IDENT)_\d+|REDACTED_(?:SECRET|URL|PATH|MODULE|IDENT)"
)

_GENERIC = set(keyword.kwlist) | set(dir(builtins))

try:
    _GENERIC |= set(sys.stdlib_module_names)
except AttributeError:  # pragma: no cover - Python <3.10 compatibility
    pass

_GENERIC = {item.lower() for item in _GENERIC}

_GENERIC |= {
    "self", "cls", "args", "kwargs", "value", "values", "result", "results", "return",
    "object", "string", "number", "default", "none", "true", "false", "import", "class",
    "method", "function", "module", "package", "test", "tests", "assert", "error", "errors",
    "exception", "message", "output", "input", "context", "request", "response", "data",
    "config", "settings", "options", "params", "field", "fields", "model", "models", "query",
    "queryset", "instance", "instances", "should", "would", "could", "which", "where", "while",
    "there", "their", "about", "these", "those", "every", "using", "called", "create", "delete",
    "update", "append", "format", "length", "python", "version", "example", "command", "print",
    "index", "count", "items", "keys",
}

_CODEY = re.compile(r"[=(){}\[\];:]|->|::|:=|\bdef\b|\bclass\b|\bimport\b|\breturn\b|\bself\b")

_LEXEME = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|==|!=|<=|>=|->|::|:=|\*\*|//|[^\s]"
)


# A long token counts as an identifier only when it is not an ordinary English word (words of eight
# letters or more, from the system dictionary); shorter identifiers must be shaped like code
# (underscore, camel case).
ENGLISH_WORDS: frozenset[str] = frozenset(
    (Path(__file__).resolve().parent / "english_words_min8.txt").read_text().split())


def _is_hient(token: str) -> bool:
    if len(token) < 16:
        return False
    return sum(bool(re.search(pattern, token)) for pattern in (r"[a-z]", r"[A-Z]", r"[0-9]", r"[_+/=-]")) >= 3



def _mask_ident(token: str) -> bool:
    if token.lower() in _GENERIC or _RX_PLACEHOLDER.fullmatch(token):
        return False
    if "_" in token.strip("_") or re.search(r"[a-z][A-Z]", token):
        return True
    if len(token) < 8:
        return False
    return token.lower() not in ENGLISH_WORDS


def extract_p0_spans(text: str) -> set[str]:
    """Hard-secret spans (P0) in `text`."""
    spans: set[str] = set()
    for regex in (_RX_KEYVAL, _RX_URL, _RX_POSIX_PATH, _RX_WINDOWS_PATH):
        spans.update(match.group(0).strip() for match in regex.finditer(text or ""))
    for match in _RX_HIENT.finditer(text or ""):
        token = match.group(0)
        if _is_hient(token):
            spans.add(token)
    return {span for span in spans if span}


def _strip_line_comment(line: str) -> str:
    if re.match(r"^\s*(?:#|//)", line):
        return ""
    line = re.sub(r"\s+#.*$", "", line)
    line = re.sub(r"\s+//.*$", "", line)
    return line.strip()


def normalized_lexemes(text: str) -> list[str]:
    """Lexemes of `text` after comment stripping (the OCO side of the transmitted text)."""
    tokens: list[str] = []
    for raw_line in (text or "").splitlines():
        tokens.extend(_LEXEME.findall(_strip_line_comment(raw_line)))
    return tokens


def private_source_tokens(text: str) -> list[str]:
    """Lexemes of the lines that look like source code (the OCO reference)."""
    tokens: list[str] = []
    for raw_line in (text or "").splitlines():
        line = _strip_line_comment(raw_line)
        if not line or line.startswith("[") or not _CODEY.search(line):
            continue
        tokens.extend(_LEXEME.findall(line))
    return tokens


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[index:index + n]) for index in range(len(tokens) - n + 1)}


class Pseudonymizer:
    def __init__(self) -> None:
        self.mapping: dict[str, str] = {}
        self._seen: dict[str, str] = {}
        self._counters = {"SECRET": 0, "URL": 0, "PATH": 0, "MODULE": 0, "IDENT": 0}

    def _replacement(self, kind: str):
        def replace(match: re.Match[str]) -> str:
            token = match.group(0)
            if _RX_PLACEHOLDER.fullmatch(token):
                return token
            if kind == "SECRET" and not (_RX_KEYVAL.fullmatch(token) or _is_hient(token)):
                return token
            if kind == "IDENT" and not _mask_ident(token):
                return token
            if token in self._seen:
                return self._seen[token]
            self._counters[kind] += 1
            placeholder = f"{kind}_{self._counters[kind]}"
            self._seen[token] = placeholder
            self.mapping[placeholder] = token
            return placeholder
        return replace

    def transform(self, text: str, *, include_p1: bool = True) -> str:
        """Pseudonymize hard secrets, URLs and paths always; module and identifier names when include_p1."""
        output = text or ""
        output = _RX_KEYVAL.sub(self._replacement("SECRET"), output)
        output = _RX_URL.sub(self._replacement("URL"), output)
        output = _RX_POSIX_PATH.sub(self._replacement("PATH"), output)
        output = _RX_WINDOWS_PATH.sub(self._replacement("PATH"), output)
        output = _RX_HIENT.sub(self._replacement("SECRET"), output)
        if not include_p1:
            return output
        output = _RX_DOTTED.sub(self._replacement("MODULE"), output)
        return _RX_IDENT.sub(self._replacement("IDENT"), output)


def restore_advice(advice: str, reverse_map: dict[str, str]) -> str:
    """Map placeholders in the expert's reply back to the original strings (host side only)."""
    if not advice or not reverse_map:
        return advice
    pattern = re.compile("|".join(re.escape(key) for key in sorted(reverse_map, key=len, reverse=True)))
    return pattern.sub(lambda match: reverse_map[match.group(0)], advice)

