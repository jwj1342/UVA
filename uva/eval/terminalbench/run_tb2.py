#!/usr/bin/env python3
"""Run Terminal-Bench 2.0 tasks under Apptainer, with any consultation policy.

Per task this reproduces the official harness's two phases with the same student and the same
consultation agent as on SWE-bench:
  agent phase    the configured agent (uva.agent.ask_probe.AskProbeAgent for consulting policies,
                 DefaultAgent for Solo) in a writable sandbox built from the task image, task =
                 instruction.md, cwd = the image's WORKDIR;
  verify phase   the task's own tests/test.sh, copied verbatim into the sandbox at /tests and run
                 there; the reward is read from /logs/verifier/reward.txt.

Budget convention: step_limit x per-command timeout (as on SWE-bench), plus a wall-clock safety
valve per task that is not part of the budget. Outputs per task under --output/<task>/:
<task>.traj.json, verifier.log, verifier/, result.json; an aggregate eval_results.json at the
root ({task: {resolved, reward, harness_error, n_asks, ask_steps, ...}}).

Prerequisites: task metadata (fetch_tb2_tasks.sh) and task images as SIFs (build_tb2_images.sh).
Verifiers install their own test dependencies from the network; forward proxy variables if needed.

    python -m uva.eval.terminalbench.run_tb2 --tasks-dir data/terminalbench2 --sif-dir containers/tb2 \
        --config configs/eval/tb2_natural.yaml --output output/runs/tb2_natural --workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import shutil
import subprocess
import threading
import time
import tomllib
from pathlib import Path

import yaml

from minisweagent.agents import get_agent_class
from minisweagent.agents.default import DefaultAgent
from uva.harness.singularity import SingularityEnvironment
from minisweagent.exceptions import LimitsExceeded
from minisweagent.models import get_model

PROXY_KEYS = ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY"]
_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


class TruncatingEnv(SingularityEnvironment):
    """Caps each observation to head + tail halves, so one chatty command cannot exhaust the context."""

    def __init__(self, *, max_observation_chars: int = 8000, **kwargs):
        self._max_obs = max_observation_chars
        super().__init__(**kwargs)

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        out = super().execute(action, cwd, timeout=timeout)
        s = out.get("output", "")
        if len(s) > self._max_obs:
            half = self._max_obs // 2
            out["output"] = (s[:half]
                             + f"\n[... output truncated: showing {self._max_obs}"
                             + f" of {len(s)} chars; use head/tail/grep ...]\n"
                             + s[-half:])
        return out


def make_agent_class(spec: str) -> type:
    """The configured agent with a wall-clock safety valve (not part of the budget)."""
    base = get_agent_class(spec) if spec else DefaultAgent

    class _Ceiling(base):  # type: ignore[valid-type,misc]
        def __init__(self, *args, deadline: float = 0.0, **kwargs):
            super().__init__(*args, **kwargs)
            self._deadline = deadline

        def query(self):
            if self._deadline and time.monotonic() > self._deadline:
                raise LimitsExceeded(
                    {"role": "exit", "content": "SafetyCeiling",
                     "extra": {"exit_status": "SafetyCeiling", "submission": ""}})
            return super().query()

    _Ceiling.__name__ = f"SafetyCeiling{base.__name__}"
    return _Ceiling



def parse_workdir(dockerfile: Path) -> str:
    """Last WORKDIR of the task image's Dockerfile; '/' when absent or variable."""
    workdir = "/"
    if dockerfile.exists():
        for line in dockerfile.read_text(errors="replace").splitlines():
            m = re.match(r"\s*WORKDIR\s+(\S+)", line)
            if m and not m.group(1).startswith(("$", "{")):
                workdir = m.group(1).strip("\"'")
    return workdir


def load_task(task_dir: Path) -> dict:
    meta = tomllib.loads((task_dir / "task.toml").read_text())
    return {
        "name": task_dir.name,
        "instruction": (task_dir / "instruction.md").read_text(),
        "workdir": parse_workdir(task_dir / "environment" / "Dockerfile"),
        "agent_timeout": float(meta.get("agent", {}).get("timeout_sec", 900.0)),
        "verifier_timeout": float(meta.get("verifier", {}).get("timeout_sec", 900.0)),
        "task_env": {k: str(v) for k, v in meta.get("environment", {}).get("env", {}).items()},
        "verifier_env": {k: str(v) for k, v in meta.get("verifier", {}).get("env", {}).items()},
        "tests_dir": task_dir / "tests",
    }


def run_verifier(env: SingularityEnvironment, task: dict, out_dir: Path) -> tuple[float | None, int]:
    """Copy tests/ into the sandbox, run test.sh there, read the reward. -> (reward, returncode)"""
    sandbox = Path(env.sandbox_dir)
    tests_dst = sandbox / "tests"
    shutil.rmtree(tests_dst, ignore_errors=True)
    shutil.copytree(task["tests_dir"], tests_dst)
    (sandbox / "logs" / "verifier").mkdir(parents=True, exist_ok=True)

    cfg = env.config
    cmd = [cfg.executable, *cfg.global_args, "exec", *cfg.exec_args]
    for key, value in {**{k: v for k, v in env_proxy_items()}, **task["verifier_env"]}.items():
        cmd.extend(["--env", f"{key}={value}"])
    if task["workdir"] != "/":
        cmd.extend(["--pwd", task["workdir"]])
    cmd.extend(["--writable", str(sandbox), "bash", "/tests/test.sh"])

    try:
        proc = subprocess.run(cmd, text=True, encoding="utf-8", errors="replace",
                              timeout=task["verifier_timeout"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out_dir / "verifier.log").write_text(proc.stdout or "")
        rc = proc.returncode
    except subprocess.TimeoutExpired as e:
        (out_dir / "verifier.log").write_text((e.output or "") + "\n[driver] verifier timeout")
        rc = -9

    reward = None
    reward_file = sandbox / "logs" / "verifier" / "reward.txt"
    if reward_file.exists():
        try:
            reward = float(reward_file.read_text().strip())
        except ValueError:
            reward = None
    ver_out = out_dir / "verifier"
    shutil.rmtree(ver_out, ignore_errors=True)
    shutil.copytree(sandbox / "logs" / "verifier", ver_out, dirs_exist_ok=True)
    return reward, rc


def env_proxy_items() -> list[tuple[str, str]]:
    import os
    return [(k, os.environ[k]) for k in PROXY_KEYS if os.environ.get(k)]


def process_task(task_dir: Path, sif: Path, out_root: Path, config: dict,
                 safety_seconds: float = 7200.0) -> dict:
    task = load_task(task_dir)
    name = task["name"]
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"instance_id": name, "resolved": False, "reward": None,
              "exit_status": "", "harness_error": False,
              "agent_steps": 0, "step_limit": config.get("agent", {}).get("step_limit", 0),
              "agent_seconds": 0.0, "verifier_returncode": None}

    env = None
    try:
        env_cfg = dict(config.get("environment", {}))
        env_cfg["image"] = str(sif)
        env_cfg["cwd"] = task["workdir"]
        env_cfg["env"] = {**env_cfg.get("env", {}), **task["task_env"]}
        env = TruncatingEnv(**env_cfg)

        model = get_model(config=config.get("model", {}))
        agent_cfg = dict(config.get("agent", {}))
        AgentClass = make_agent_class(agent_cfg.pop("agent_class", ""))
        agent = AgentClass(model, env,
                           deadline=time.monotonic() + safety_seconds,
                           **agent_cfg)
        t0 = time.monotonic()
        try:
            info = agent.run(task["instruction"])
            result["exit_status"] = info.get("exit_status", "")
        except Exception as e:  # noqa: BLE001 -- state may still pass verification
            result["exit_status"] = type(e).__name__
        result["agent_seconds"] = round(time.monotonic() - t0, 1)
        result["agent_steps"] = agent.n_calls
        asks = getattr(agent, "asks", [])
        result["n_asks"] = len(asks)
        result["ask_steps"] = [a.get("step") for a in asks]
        agent.save(out_dir / f"{name}.traj.json", {"info": {"instance_id": name}})

        reward, rc = run_verifier(env, task, out_dir)
        result["reward"] = reward
        result["verifier_returncode"] = rc
        result["resolved"] = reward is not None and reward >= 0.999
        if reward is None:
            result["harness_error"] = rc != 0  # no reward written at all
    except Exception as e:  # sandbox build / driver failure = infrastructure
        result["exit_status"] = result["exit_status"] or type(e).__name__
        result["harness_error"] = True
        (out_dir / "driver_error.log").write_text(f"{type(e).__name__}: {e}")
    finally:
        if env is not None:
            env.cleanup()

    (out_dir / "result.json").write_text(json.dumps(result, indent=1))
    log(f"[{name}] resolved={result['resolved']} reward={result['reward']} "
        f"exit={result['exit_status']} agent={result['agent_seconds']}s")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", required=True)
    ap.add_argument("--sif-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--slice", default="", help="i:j over the sorted task list")
    ap.add_argument("--only", default="", help="comma-separated task names")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--safety-seconds", type=float, default=7200.0,
                    help="wall-clock safety valve per task; NOT the budget (steps are)")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    task_dirs = sorted(p.parent for p in Path(args.tasks_dir).glob("*/task.toml"))
    if args.only:
        wanted = set(args.only.split(","))
        task_dirs = [d for d in task_dirs if d.name in wanted]
    if args.slice:
        i, j = args.slice.split(":")
        task_dirs = task_dirs[int(i):int(j)]
    # longest agent timeout first, so the wall-clock tail is not one huge task
    task_dirs.sort(key=lambda d: -load_task(d)["agent_timeout"])

    todo = []
    for d in task_dirs:
        sif = Path(args.sif_dir) / f"{d.name}.sif"
        if not args.redo and (out_root / d.name / "result.json").exists():
            log(f"[{d.name}] skip (done)")
            continue
        if not sif.exists():
            log(f"[{d.name}] skip (no SIF yet)")
            continue
        todo.append((d, sif))
    log(f"=== {len(todo)} tasks to run, workers={args.workers} ===")

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_task, d, sif, out_root, config, args.safety_seconds): d.name
                for d, sif in todo}
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            results[r["instance_id"]] = r

    # merge with anything already on disk, then write the aggregate
    for rj in out_root.glob("*/result.json"):
        r = json.loads(rj.read_text())
        results.setdefault(r["instance_id"], r)
    (out_root / "eval_results.json").write_text(json.dumps(
        {k: results[k] for k in sorted(results)}, indent=1))
    n = len(results)
    solved = sum(r["resolved"] for r in results.values())
    herr = sum(r["harness_error"] for r in results.values())
    log(f"=== done: {solved}/{n} resolved, {herr} harness_error -> {out_root}/eval_results.json ===")


if __name__ == "__main__":
    main()
