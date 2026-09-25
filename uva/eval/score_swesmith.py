#!/usr/bin/env python3
"""Two-sided SWE-smith scorer (Apptainer). S(tau) for the training tasks.

SWE-smith bakes each instance's bug into a git branch named after the instance inside the
repository image, and its tests are the golden ones on the clean `main` branch. Per instance:
check out the bug branch, apply the model patch, restore the golden test files from `main`, run
FAIL_TO_PASS (and PASS_TO_PASS), and then -- on every cell whose FAIL_TO_PASS all pass -- reset
to the unpatched bug branch and run FAIL_TO_PASS again. A cell is resolved only if the tests pass
WITH the patch and do NOT all pass WITHOUT it; otherwise it is a phantom (the task was never
broken as far as its tests can tell) and nothing the agent did can be credited.

Fields per instance: resolved, f2p_resolved, harness_error, patch_applied, phantom,
baseline_f2p_pass / baseline_f2p_total / baseline_f2p_run_missing, p2p_pass / p2p_total, ...

    python -m uva.eval.score_swesmith --preds RUN/preds.json --records data/pools/swesmith_train.jsonl \\
        --output RUN/eval_results.json --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from uva.eval.score_verified import SIF_CACHE_DIR, apply_patch, build_sandbox, exec_sandbox, parse_test_results

# any conda env python first (the images ship the repo's env as a conda env), then fallbacks
_PY_PROBE = (
    r'for p in $(ls /opt/miniconda3/envs/*/bin/python /opt/conda/envs/*/bin/python 2>/dev/null) '
    r'/opt/miniconda3/bin/python /usr/local/bin/python /usr/bin/python3 /usr/bin/python; do '
    r'[ -x "$p" ] && { echo "$p"; break; }; done'
)


def sif_for_image(image_name: str) -> Path:
    base = image_name.rsplit("/", 1)[-1].split(":")[0]
    return SIF_CACHE_DIR / f"{base}.sif"


def detect_python(sandbox) -> str:
    rc, out = exec_sandbox(sandbox, _PY_PROBE, timeout=60)
    for ln in out.splitlines():
        ln = ln.strip()
        if ln.startswith("/") and ln.endswith("python"):
            return ln
    return ""


def restore_golden_tests(sandbox, tests: list) -> None:
    """The bug branch may mutate/remove test files; the golden ones live on `main`."""
    files = sorted({t.split("::")[0] for t in tests if "::" in t})
    if files:
        q = " ".join(shlex.quote(f) for f in files)
        exec_sandbox(sandbox, f"cd /testbed && git checkout main -- {q} 2>&1", timeout=180)


def pytest_cmd(py: str, tests: list) -> str | None:
    files = sorted({t.split("::")[0] for t in tests if "::" in t})
    if not files:
        return None
    targets = " ".join(shlex.quote(f) for f in files)
    return f"cd /testbed && {py} -m pytest {targets} --tb=no -v --no-header -rN 2>&1"


def classify_f2p_statuses(statuses: dict[str, str], *, baseline_statuses: dict[str, str] | None = None) -> dict:
    """Separate infrastructure misses from collection failures caused by a patch."""
    f2p_pass = sum(1 for s in statuses.values() if s == "pass")
    f2p_missing = sum(1 for s in statuses.values() if s == "missing")
    baseline_f2p_missing = None
    agent_test_error = False
    harness_error = f2p_missing > 0
    if f2p_missing and baseline_statuses is not None:
        baseline_f2p_missing = sum(1 for s in baseline_statuses.values() if s == "missing")
        if baseline_f2p_missing == 0:
            harness_error = False
            agent_test_error = True
    return {"f2p_pass": f2p_pass, "f2p_missing": f2p_missing,
            "f2p_resolved": (not harness_error and not agent_test_error and f2p_pass == len(statuses)),
            "harness_error": harness_error, "agent_test_error": agent_test_error,
            "baseline_f2p_missing": baseline_f2p_missing}


def _run_f2p(sandbox, py: str, f2p: list, timeout: int) -> tuple[dict, str]:
    _, out = exec_sandbox(sandbox, pytest_cmd(py, f2p), timeout=timeout)
    return parse_test_results(out, f2p), out


def _reset_to_bug_branch(sandbox, iid: str, f2p: list) -> tuple[bool, str]:
    rc, out = exec_sandbox(sandbox, f"cd /testbed && git reset --hard {shlex.quote(iid)} && git clean -fd 2>&1",
                           timeout=180)
    if rc != 0:
        return False, out
    restore_golden_tests(sandbox, f2p)
    return True, out


def evaluate_instance(rec: dict, prediction: str, f2p_only: bool, f2p_timeout: int = 600) -> dict:
    iid = rec["instance_id"]
    t0 = time.time()

    def fail(msg: str, **extra) -> dict:
        d = {"instance_id": iid, "resolved": False, "f2p_resolved": False, "harness_error": True,
             "error": msg, "phantom": False, "twosided": True, "elapsed": round(time.time() - t0, 1)}
        d.update(extra)
        return d

    sif = sif_for_image(rec.get("image_name", ""))
    if not sif.is_file():
        return fail(f"SIF not found: {sif.name}")
    f2p = rec["FAIL_TO_PASS"]
    p2p = rec.get("PASS_TO_PASS", [])
    if isinstance(f2p, str):
        f2p = json.loads(f2p)
    if isinstance(p2p, str):
        p2p = json.loads(p2p)
    if not f2p:
        return fail("no FAIL_TO_PASS")

    sandbox = None
    try:
        sandbox = build_sandbox(sif)
        rc, out = exec_sandbox(sandbox, f"cd /testbed && git checkout -f {shlex.quote(iid)} 2>&1", timeout=120)
        if rc != 0:
            return fail(f"git checkout {iid} failed: {out[-200:]}")
        py = detect_python(sandbox)
        if not py:
            return fail("no python interpreter found in image")
        if pytest_cmd(py, f2p) is None:
            return fail("no runnable f2p target (bare ids?)")

        # side 1: with the model patch
        prediction = prediction or ""
        patch_ok, _ = apply_patch(sandbox, prediction, "model_patch") if prediction.strip() else (False, "")
        restore_golden_tests(sandbox, f2p if f2p_only else (f2p + p2p))
        f2p_st, out_f2p = _run_f2p(sandbox, py, f2p, f2p_timeout)
        f2p_result = classify_f2p_statuses(f2p_st)
        baseline_tail = ""
        baseline_pass = None
        baseline_run_missing = None
        phantom = False

        # side 2: without it, on exactly the cells that need it
        need_baseline = (f2p_result["f2p_pass"] == len(f2p)) or (
            f2p_result["f2p_missing"] and patch_ok and prediction.strip())
        if not prediction.strip():
            baseline_pass = f2p_result["f2p_pass"]
            phantom = baseline_pass == len(f2p)
            need_baseline = False
        if need_baseline:
            ok, reset_out = _reset_to_bug_branch(sandbox, iid, f2p)
            if ok:
                base_st, base_out = _run_f2p(sandbox, py, f2p, f2p_timeout)
                baseline_tail = base_out[-300:]
                baseline_pass = sum(1 for s in base_st.values() if s == "pass")
                baseline_run_missing = sum(1 for s in base_st.values() if s == "missing")
                phantom = baseline_pass == len(f2p)
                if f2p_result["f2p_missing"]:
                    f2p_result = classify_f2p_statuses(f2p_st, baseline_statuses=base_st)
            else:
                return fail(f"baseline reset failed: {reset_out[-160:]}", patch_applied=patch_ok,
                            f2p_pass=f2p_result["f2p_pass"], f2p_total=len(f2p))

        f2p_resolved = bool(f2p_result["f2p_resolved"]) and not phantom

        p2p_pass = p2p_missing = 0
        out_p2p = ""
        if p2p and not f2p_only and f2p_resolved:
            if need_baseline:  # back to the patched state
                exec_sandbox(sandbox, f"cd /testbed && git checkout -f {shlex.quote(iid)} 2>&1", timeout=120)
                if prediction.strip():
                    apply_patch(sandbox, prediction, "model_patch")
                restore_golden_tests(sandbox, f2p + p2p)
            cmd2 = pytest_cmd(py, p2p)
            if cmd2:
                _, out_p2p = exec_sandbox(sandbox, cmd2, timeout=900)
                p2p_st = parse_test_results(out_p2p, p2p)
                p2p_pass = sum(1 for s in p2p_st.values() if s == "pass")
                p2p_missing = sum(1 for s in p2p_st.values() if s == "missing")
        resolved = f2p_resolved and (f2p_only or (p2p_missing == 0 and p2p_pass == len(p2p)))
        return {
            "instance_id": iid, "resolved": resolved, "f2p_resolved": f2p_resolved,
            "harness_error": bool(f2p_result["harness_error"]), "patch_applied": patch_ok, "python": py,
            "f2p_pass": f2p_result["f2p_pass"], "f2p_total": len(f2p), "f2p_missing": f2p_result["f2p_missing"],
            "agent_test_error": f2p_result["agent_test_error"],
            "baseline_f2p_missing": f2p_result["baseline_f2p_missing"],
            "baseline_f2p_run_missing": baseline_run_missing,
            "baseline_f2p_pass": baseline_pass, "baseline_f2p_total": len(f2p),
            "phantom": phantom, "twosided": True,
            "p2p_pass": p2p_pass, "p2p_total": len(p2p), "p2p_skipped": bool(f2p_only),
            "elapsed": round(time.time() - t0, 1),
            "test_output_tail": (out_f2p + out_p2p)[-600:], "baseline_output_tail": baseline_tail,
        }
    except Exception as e:  # noqa: BLE001
        return fail(str(e)[:300])
    finally:
        if sandbox:
            shutil.rmtree(sandbox, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--records", required=True, help="pool jsonl (instance_id, image_name, FAIL_TO_PASS, PASS_TO_PASS)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=int(os.getenv("EVAL_WORKERS", "2")))
    ap.add_argument("--f2p-timeout", type=int, default=600)
    ap.add_argument("--f2p-only", action="store_true")
    args = ap.parse_args()

    preds = json.loads(Path(args.preds).read_text())
    recs = {}
    for line in Path(args.records).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["instance_id"] in preds:
                recs[r["instance_id"]] = r
    print(f"{len(preds)} preds; matched {len(recs)} records", flush=True)
    results: dict = {}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    def proc(iid):
        rec = recs.get(iid)
        if rec is None:
            return {"instance_id": iid, "resolved": False, "f2p_resolved": False, "harness_error": True,
                    "error": "not in records", "phantom": False, "twosided": True}
        return evaluate_instance(rec, preds[iid].get("model_patch", "") or "", args.f2p_only, args.f2p_timeout)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(proc, i): i for i in preds}
        done = 0
        for f in as_completed(futs):
            r = f.result()
            results[r["instance_id"]] = r
            done += 1
            out.write_text(json.dumps(results, indent=2))
            icon = "ok" if r.get("f2p_resolved") else ("phantom" if r.get("phantom") else
                                                       ("harness" if r.get("harness_error") else "fail"))
            print(f"[{done}/{len(preds)}] {icon:7} {r['instance_id']} f2p={r.get('f2p_pass', 0)}/{r.get('f2p_total', 0)} "
                  f"base={r.get('baseline_f2p_pass')} ({r.get('elapsed', 0):.0f}s)"
                  f"{'  ' + r['error'] if r.get('error') else ''}", flush=True)
    nres = sum(1 for r in results.values() if r.get("resolved"))
    nph = sum(1 for r in results.values() if r.get("phantom"))
    nherr = sum(1 for r in results.values() if r.get("harness_error"))
    print(f"\nresolved {nres}/{len(results)}  phantom {nph}  harness_error {nherr}  -> {out}", flush=True)


if __name__ == "__main__":
    main()
