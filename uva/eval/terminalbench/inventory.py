"""Per-task inventory for Terminal-Bench: the reference r is the task's environment definition
(Dockerfile, task.toml, compose files, everything under environment/), and I(r) holds its
host-specific identifiers: absolute paths under the task's directories, non-default ports, service /
container / volume names, and declared environment variables. Standard tool names, default ports and
ordinary words are never charged. L_P1 and OCO are then computed as on repair tasks.
"""
from __future__ import annotations

import re
from pathlib import Path

from uva.privacy.detector import ENGLISH_WORDS

_DEFAULT_PORTS = {"20", "21", "22", "23", "25", "53", "80", "110", "143", "443", "993", "995", "3306",
                  "5432", "6379", "8080", "8000", "27017"}
_SYSTEM_ROOTS = ("/usr", "/bin", "/sbin", "/lib", "/etc", "/tmp", "/dev", "/proc", "/sys", "/var/lib/apt",
                 "/root/.cache", "/opt/conda")
_PATH = re.compile(r"(?<![\w/])(/[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+)")
_PORT = re.compile(r"(?:EXPOSE\s+|(?:^|[\s\"':=])(?:port|ports)[\"']?\s*[:=]\s*[\"'\[]*\s*|:)(\d{2,5})\b", re.I | re.M)
_ENV_TOML = re.compile(r"^\s*([A-Z][A-Z0-9_]{2,})\s*=", re.M)
_ENV_DOCKER = re.compile(r"^\s*(?:ENV|ARG)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)
_SERVICE = re.compile(r"^\s{2,}([a-z][a-z0-9_\-]{2,}):\s*$", re.M)          # compose service keys
_NAMED = re.compile(r"^\s*(?:container_name|volume_name|hostname|name):\s*[\"']?([A-Za-z0-9_.\-]{3,})", re.M)
_VOLUME = re.compile(r"^\s*-\s*([A-Za-z0-9_\-]{3,}):/", re.M)


def _is_identifier_like(tok: str) -> bool:
    return bool(re.search(r"[_.\d]|[a-z][A-Z]", tok)) or (len(tok) >= 8 and tok.lower() not in ENGLISH_WORDS)


def task_reference(task_dir: Path) -> tuple[set[str], str]:
    """(I(r), r) for one task directory: the inventory and the environment-definition text."""
    files = []
    env_dir = task_dir / "environment"
    for f in sorted(env_dir.rglob("*")) if env_dir.exists() else []:
        if f.is_file() and f.stat().st_size < 512_000:
            files.append(f)
    for name in ("task.toml", "docker-compose.yaml", "docker-compose.yml", "compose.yaml"):
        if (task_dir / name).is_file():
            files.append(task_dir / name)
    text = "\n".join(f.read_text(errors="replace") for f in files)
    inv: set[str] = set()
    for m in _PATH.finditer(text):
        path = m.group(1).rstrip("/.,;:)\"'")
        if path.startswith(_SYSTEM_ROOTS) or path.count("/") < 2:
            continue
        inv.add(path)
        base = path.rsplit("/", 1)[-1]
        if len(base) > 3 and _is_identifier_like(base):
            inv.add(base)
    for m in _PORT.finditer(text):
        if m.group(1) not in _DEFAULT_PORTS and 1024 <= int(m.group(1)) <= 65535:
            inv.add(m.group(1))
    for rx in (_ENV_TOML, _ENV_DOCKER):
        for m in rx.finditer(text):
            if m.group(1) not in ("PATH", "HOME", "USER", "SHELL", "LANG", "TERM", "DEBIAN_FRONTEND", "PYTHONUNBUFFERED"):
                inv.add(m.group(1))
    for rx in (_SERVICE, _NAMED, _VOLUME):
        for m in rx.finditer(text):
            tok = m.group(1)
            if tok not in ("services", "volumes", "networks", "environment", "build", "image", "ports", "command",
                           "depends_on", "healthcheck", "agent", "verifier", "metadata") and _is_identifier_like(tok):
                inv.add(tok)
    return inv, text


def load_inventories(tasks_dir: Path) -> dict[str, tuple[set[str], str]]:
    return {d.parent.name: task_reference(d.parent) for d in sorted(Path(tasks_dir).glob("*/task.toml"))}


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="print the inventory of one or every task")
    ap.add_argument("tasks_dir")
    ap.add_argument("--task", default="")
    a = ap.parse_args()
    for name, (inv, _) in load_inventories(Path(a.tasks_dir)).items():
        if not a.task or name == a.task:
            print(json.dumps({"task": name, "inventory_size": len(inv), "inventory": sorted(inv)}))
