#!/usr/bin/env python3
"""Local SWE-bench Verified scorer (Apptainer, official-compatible logic).

For each prediction: build a writable sandbox from the instance's evaluation image, apply the
benchmark's test patch, apply the model patch, run FAIL_TO_PASS and PASS_TO_PASS in separate
invocations, and grade per test id from verbose output. Fields per instance:

    resolved       all FAIL_TO_PASS pass AND all PASS_TO_PASS pass (the strict rule reported as pass@k)
    f2p_resolved   all FAIL_TO_PASS pass (the primary utility signal on this local grader)
    harness_error  a FAIL_TO_PASS id never appeared in the output (selection/harness problem, not an
                   agent failure); such instances count as failures in every reported rate
    patch_applied  whether the model patch applied

Images: sweb.eval.x86_64.<instance>.sif under UVA_SIF_CACHE_DIR (pulled from Docker Hub when absent).
The `preds.json` produced by the harness can equally be scored with the official SWE-bench harness.

    python -m uva.eval.score_verified --preds output/runs/verified_solo/preds.json \
        --output output/runs/verified_solo/eval_results.json --workers 8
"""

import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import os

SIF_CACHE_DIR = Path(os.getenv("UVA_SIF_CACHE_DIR", os.getenv("MSWEA_SIF_CACHE_DIR", "containers")))
APPTAINER = os.getenv("MSWEA_SINGULARITY_EXECUTABLE", "apptainer")


# ---------------------------------------------------------------------------
# Container helpers
# ---------------------------------------------------------------------------

def get_sif_path(instance_id: str) -> Path | None:
    """Existing SIF path (original or lowercased image name) or None."""
    id_docker = instance_id.replace("__", "_1776_")
    p = SIF_CACHE_DIR / f"sweb.eval.x86_64.{id_docker}.sif"
    if p.exists():
        return p
    p_lower = SIF_CACHE_DIR / f"sweb.eval.x86_64.{id_docker.lower()}.sif"
    if p_lower.exists():
        return p_lower
    return None


def get_sif_or_pull(instance_id: str, dataset: str) -> Path:
    """Cached SIF path; pull from Docker Hub once if absent."""
    cached = get_sif_path(instance_id)
    if cached is not None:
        return cached

    id_docker = instance_id.replace("__", "_1776_")
    is_rebench = "rebench" in dataset.lower()
    if is_rebench:
        registry = "swerebench"
        tag_key = id_docker.lower()
    else:
        registry = "swebench"
        tag_key = id_docker
    sif_name = f"sweb.eval.x86_64.{tag_key}.sif"
    target = SIF_CACHE_DIR / sif_name
    target_tmp = SIF_CACHE_DIR / f"{sif_name}.tmp.{uuid.uuid4().hex[:6]}"
    docker_url = f"docker://{registry}/sweb.eval.x86_64.{tag_key}:latest"

    SIF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [APPTAINER, "build", str(target_tmp), docker_url],
        capture_output=True, text=True, timeout=900,
    )
    if r.returncode != 0:
        if target_tmp.exists():
            target_tmp.unlink()
        raise RuntimeError(f"docker pull failed for {docker_url}: {r.stderr[:300]}")
    target_tmp.rename(target)
    return target


def build_sandbox(sif_path: Path) -> Path:
    sandbox_dir = Path(tempfile.gettempdir()) / f"sweval-{uuid.uuid4().hex[:8]}"
    r = subprocess.run(
        [APPTAINER, "build", "--sandbox", str(sandbox_dir), str(sif_path)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        shutil.rmtree(sandbox_dir, ignore_errors=True)
        raise RuntimeError(f"sandbox build failed: {r.stderr[:400]}")
    for d in ["project", "scratch", "localscratch", "home"]:
        (sandbox_dir / d).mkdir(exist_ok=True)
    return sandbox_dir


# `--cleanenv` starts the container under the C locale; the official images run under UTF-8.
_SANDBOX_ENV = (
    "LC_ALL=C.UTF-8",
    "LANG=C.UTF-8",
    "PYTHONIOENCODING=utf-8",
)


def exec_sandbox(sandbox_dir: Path, command: str, timeout: int = 300) -> tuple[int, str]:
    env_flags = [flag for kv in _SANDBOX_ENV for flag in ("--env", kv)]
    r = subprocess.run(
        [APPTAINER, "--quiet", "exec",
         "--contain", "--cleanenv", "--fakeroot", "--writable",
         *env_flags,
         str(sandbox_dir), "bash", "-c", command],
        capture_output=True, text=True,
        timeout=timeout, encoding="utf-8", errors="replace",
    )
    return r.returncode, r.stdout + r.stderr


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------

def apply_patch(sandbox_dir: Path, patch_text: str, label: str) -> tuple[bool, str]:
    """Apply a patch inside the sandbox with git apply. Returns (applied, output)."""
    if not patch_text or not patch_text.strip():
        return False, "empty patch"
    patch_file = sandbox_dir / "testbed" / f"{label}.diff"
    patch_file.write_text(patch_text, encoding="utf-8")
    rc, out = exec_sandbox(
        sandbox_dir,
        f"cd /testbed && git apply --whitespace=nowarn /testbed/{label}.diff && rm -f /testbed/{label}.diff",
        timeout=300,
    )
    patch_file.unlink(missing_ok=True)
    return rc == 0, out


# ---------------------------------------------------------------------------
# Test command construction
# ---------------------------------------------------------------------------

def extract_test_files_from_patch(test_patch: str) -> list[str]:
    """Return file paths added/modified by the test patch."""
    files = []
    for line in test_patch.splitlines():
        if line.startswith("diff --git "):
            # "diff --git a/path/to/file.py b/path/to/file.py"
            parts = line.split(" b/", 1)
            if len(parts) == 2:
                files.append(parts[1].strip())
    return files


TESTBED_PYTHON = "/opt/miniconda3/envs/testbed/bin/python"


_SYMPY_TOKEN_RE = re.compile(
    r"(\d+)\s+(passed|failed|skipped|expected to fail|exceptions|xpassed)", re.IGNORECASE
)


def _sympy_summary(output: str) -> tuple[int, int, int]:
    """(passed, failed, exceptions) from sympy's `tests finished:` line; (0, 999, 0) if it never ran.
    Exceptions count as failures on the name-selected FAIL_TO_PASS run and are pre-existing noise on
    a whole-file PASS_TO_PASS run, so the caller decides."""
    marker = output.rfind("tests finished:")
    summary = output[marker:] if marker >= 0 else output
    counts = {kind.lower(): int(n) for n, kind in _SYMPY_TOKEN_RE.findall(summary)}
    if "passed" not in counts and "failed" not in counts and "exceptions" not in counts:
        return 0, 999, 0  # sentinel: the runner never produced a parseable summary
    return counts.get("passed", 0), counts.get("failed", 0), counts.get("exceptions", 0)


def build_sympy_test_cmd(test_names: list[str], test_files: list[str]) -> str:
    """Build sympy bin/test command for a list of bare test names."""
    files_arg = " ".join(test_files) if test_files else "sympy"
    k_filter = "|".join(test_names)
    return (
        f'cd /testbed && {TESTBED_PYTHON} bin/test {files_arg} '
        f'-k "{k_filter}" 2>&1'
    )


def sympy_locate_test_files(sandbox_dir: Path, test_names: list[str]) -> list[str]:
    """Files defining these bare sympy test names (PASS_TO_PASS names live outside the test patch)."""
    if not test_names:
        return []
    # Escape is unnecessary for identifiers, but bound the pattern so `test_foo` cannot match
    # `test_foobar` and pull in an unrelated file.
    alternation = "|".join(sorted(set(test_names)))
    rc, out = exec_sandbox(
        sandbox_dir,
        f"cd /testbed && grep -rlE 'def ({alternation})\\(' --include='*.py' . 2>/dev/null | sed 's|^\\./||'",
        timeout=300,
    )
    if rc != 0 and not out.strip():
        return []
    files = [line.strip() for line in out.splitlines() if line.strip().endswith(".py")]
    # Keep it to the test tree; a name can also appear in docs or generation helpers.
    return sorted({f for f in files if "test" in Path(f).name})


def _django_dotted(test_id: str) -> str:
    """`test_X (module.Class)` -> `module.Class.test_X`."""
    m = re.match(r"^(\w+)\s*\(([\w.]+)\)\s*$", test_id.strip())
    return f"{m.group(2)}.{m.group(1)}" if m else test_id


def _django_module(test_id: str) -> str:
    """`test_X (auth_tests.test_validators.Cls)` -> `auth_tests.test_validators`."""
    m = re.match(r"^\w+\s*\(([\w.]+)\)\s*$", test_id.strip())
    if not m:
        return test_id
    parts = m.group(1).split(".")
    while len(parts) > 1 and parts[-1][:1].isupper():  # strip Class (and nested Class) components
        parts.pop()
    return ".".join(parts)


def build_test_command(instance_id: str, tests: list[str]) -> str | None:
    """Command running `tests` by file / module (not by node id), or None if no target can be derived
    (the caller then flags harness_error). sympy uses bin/test, django tests/runtests.py, others pytest."""
    if not tests:
        return None
    if instance_id.startswith("sympy__"):
        raise RuntimeError("use build_sympy_test_cmd() for sympy instances")
    if instance_id.startswith("django__"):
        modules = sorted({_django_module(t) for t in tests})
        labels = " ".join(shlex.quote(m) for m in modules)
        return (
            f"cd /testbed && {TESTBED_PYTHON} tests/runtests.py "
            f"--verbosity 2 --settings=test_sqlite --noinput {labels} 2>&1"
        )
    files = sorted({t.split("::")[0] for t in tests if "::" in t})
    if not files:  # bare ids on the pytest path → cannot reliably select → harness error
        return None
    targets_str = " ".join(shlex.quote(f) for f in files)
    return (
        f"cd /testbed && {TESTBED_PYTHON} -m pytest {targets_str} "
        f"--tb=no -v --no-header -rN 2>&1"
    )


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def parse_test_results(output: str, test_ids: list[str]) -> dict[str, str]:
    """Per-id status in {"pass", "fail", "missing"}; "missing" (never appeared in the output) is a
    harness signal, not an agent failure."""
    results: dict[str, str] = {}
    for tid in test_ids:
        m_django = re.match(r"^(\w+)\s*\(([\w.]+)\)\s*$", tid.strip())
        if m_django:
            func, mod_class = m_django.group(1), m_django.group(2)
            patterns = [
                re.escape(tid) + r"\s+\.\.\.\s+(ok|FAIL|ERROR|skipped|expected failure)",
                re.escape(func) + r"\s*\(" + re.escape(mod_class) + r"(?:\." + re.escape(func) + r")?\)\s+\.\.\.\s+(ok|FAIL|ERROR|skipped|expected failure)",
            ]
            status = "missing"
            for pat in patterns:
                m = re.search(pat, output)
                if m:
                    status = "pass" if m.group(1) in ("ok", "expected failure") else "fail"
                    break
            results[tid] = status
            continue

        if "::" in tid:
            func = re.escape(tid.split("::")[-1])
            pattern = re.escape(tid) + r"[ \t]+(PASSED|FAILED|ERROR|XPASS|XFAIL|SKIPPED)"
            alt = r"\S+::" + func + r"[ \t]+(PASSED|FAILED|ERROR|XPASS|XFAIL|SKIPPED)"
            m = re.search(pattern, output) or re.search(alt, output)
            if m:
                results[tid] = "pass" if m.group(1) in ("PASSED", "XPASS") else "fail"
            else:
                results[tid] = "missing"
        else:
            pytest_pat = r"::" + re.escape(tid) + r"[ \t]+(PASSED|FAILED|ERROR|XPASS|XFAIL|SKIPPED)"
            sympy_pat = r"(?m)^\s*" + re.escape(tid) + r"\s+(ok|FAILED|ERROR|XFAIL)"
            m = re.search(pytest_pat, output)
            if m:
                results[tid] = "pass" if m.group(1) in ("PASSED", "XPASS") else "fail"
            else:
                m = re.search(sympy_pat, output)
                results[tid] = ("pass" if m.group(1) == "ok" else "fail") if m else "missing"

    return results


# ---------------------------------------------------------------------------
# Per-instance evaluation
# ---------------------------------------------------------------------------

def evaluate_instance(instance: dict, prediction: str, dataset: str = "") -> dict:
    iid = instance["instance_id"]
    t0 = time.time()

    try:
        sif_path = get_sif_or_pull(iid, dataset)
    except Exception as e:
        return {"instance_id": iid, "resolved": False,
                "error": f"SIF pull failed: {str(e)[:200]}",
                "elapsed": round(time.time() - t0, 1)}

    f2p = instance["FAIL_TO_PASS"]
    p2p = instance["PASS_TO_PASS"]
    if isinstance(f2p, str):
        f2p = json.loads(f2p)
    if isinstance(p2p, str):
        p2p = json.loads(p2p)
    test_patch = instance.get("test_patch", "")

    sandbox_dir = None
    try:
        sandbox_dir = build_sandbox(sif_path)

        # Apply test patch
        ok, msg = apply_patch(sandbox_dir, test_patch, "test_patch")
        if not ok and test_patch.strip():
            return {"instance_id": iid, "resolved": False,
                    "error": f"test_patch failed: {msg[:200]}",
                    "elapsed": round(time.time() - t0, 1)}

        # Apply model prediction
        patch_ok, _ = apply_patch(sandbox_dir, prediction, "model_patch")

        # Run tests
        if not f2p:
            return {"instance_id": iid, "resolved": False,
                    "error": "no FAIL_TO_PASS tests", "elapsed": round(time.time() - t0, 1)}

        f2p_missing = p2p_missing = 0
        if iid.startswith("sympy__"):
            # sympy: bin/test with a count-based summary; PASS_TO_PASS files are run whole
            test_files = extract_test_files_from_patch(test_patch)
            _, out_f2p = exec_sandbox(sandbox_dir, build_sympy_test_cmd(f2p, test_files), timeout=300)
            f2p_passed, f2p_failed_n, f2p_exc = _sympy_summary(out_f2p)
            f2p_failed = f2p_failed_n if f2p_failed_n >= 999 else f2p_failed_n + f2p_exc
            f2p_pass = f2p_passed
            harness_error = (f2p_passed == 0 and f2p_failed >= 999)  # f2p never ran/parsed
            p2p_pass, output = f2p_passed, out_f2p
            p2p_ran = True
            if p2p:
                p2p_files = sympy_locate_test_files(sandbox_dir, p2p) or test_files
                if p2p_files:
                    _, out_p2p = exec_sandbox(
                        sandbox_dir,
                        f"cd /testbed && {TESTBED_PYTHON} bin/test {' '.join(p2p_files)} 2>&1",
                        timeout=900,
                    )
                    p2p_passed, p2p_failed_n, p2p_exc = _sympy_summary(out_p2p)
                    p2p_ran = not (p2p_passed == 0 and p2p_failed_n >= 999)
                    p2p_pass = p2p_passed
                    p2p_regressions = 0 if p2p_failed_n >= 999 else p2p_failed_n
                    p2p_exceptions = p2p_exc
                    p2p_missing = 0 if p2p_ran else len(p2p)
                    output = out_f2p + "\n---p2p---\n" + out_p2p
                else:
                    p2p_ran, p2p_regressions, p2p_missing = False, 0, len(p2p)
            else:
                p2p_regressions = 0
            f2p_resolved = (not harness_error) and f2p_failed == 0 and f2p_passed > 0
            resolved = f2p_resolved and (not p2p or (p2p_ran and p2p_regressions == 0))
        else:
            # FAIL_TO_PASS and PASS_TO_PASS in separate invocations
            cmd_f2p = build_test_command(iid, f2p)
            if cmd_f2p is None:
                return {"instance_id": iid, "resolved": False, "f2p_resolved": False,
                        "harness_error": "no runnable f2p target", "patch_applied": patch_ok,
                        "elapsed": round(time.time() - t0, 1)}
            _, out_f2p = exec_sandbox(sandbox_dir, cmd_f2p, timeout=600)
            f2p_st = parse_test_results(out_f2p, f2p)
            f2p_pass = sum(1 for s in f2p_st.values() if s == "pass")
            f2p_missing = sum(1 for s in f2p_st.values() if s == "missing")
            output = out_f2p
            p2p_pass = 0
            if p2p:
                cmd_p2p = build_test_command(iid, p2p)
                if cmd_p2p is not None:
                    _, out_p2p = exec_sandbox(sandbox_dir, cmd_p2p, timeout=900)
                    p2p_st = parse_test_results(out_p2p, p2p)
                    p2p_pass = sum(1 for s in p2p_st.values() if s == "pass")
                    p2p_missing = sum(1 for s in p2p_st.values() if s == "missing")
                    output = out_f2p + "\n---p2p---\n" + out_p2p[-400:]
                else:
                    p2p_missing = len(p2p)
            harness_error = (f2p_missing > 0)
            f2p_resolved = (not harness_error) and (f2p_pass == len(f2p))
            resolved = f2p_resolved and (p2p_missing == 0) and (p2p_pass == len(p2p))

        return {
            "instance_id": iid,
            "resolved": resolved,         # strict: f2p AND p2p (SWE-bench standard)
            "f2p_resolved": f2p_resolved, # primary utility metric: bug-fix tests all pass
            "harness_error": bool(harness_error),  # f2p test(s) unmatched → NOT an agent failure
            "patch_applied": patch_ok,
            "f2p_pass": f2p_pass,
            "f2p_total": len(f2p),
            "f2p_missing": f2p_missing,
            "p2p_pass": p2p_pass,
            "p2p_total": len(p2p),
            "p2p_missing": p2p_missing,
            "elapsed": round(time.time() - t0, 1),
            "test_output_tail": output[-600:],
        }

    except Exception as e:
        return {"instance_id": iid, "resolved": False,
                "error": str(e), "elapsed": round(time.time() - t0, 1)}
    finally:
        if sandbox_dir:
            shutil.rmtree(sandbox_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Local SWE-bench evaluation via Apptainer")
    parser.add_argument("--preds", required=True, help="Path to predictions JSON")
    parser.add_argument("--output", required=True, help="Path to write results JSON")
    parser.add_argument("--workers", type=int,
                        default=int(os.getenv("EVAL_WORKERS", "4")),
                        help="Parallel workers")
    parser.add_argument("--instance", help="Evaluate a single instance ID (for debugging)")
    parser.add_argument("--dataset", default=os.getenv("EVAL_DATASET", "princeton-nlp/SWE-Bench_Verified"),
                        help="HF dataset path (e.g., nebius/SWE-rebench)")
    parser.add_argument("--split", default=os.getenv("EVAL_SPLIT", "test"),
                        help="Dataset split (e.g., 'test', 'filtered')")
    args = parser.parse_args()

    preds: dict = json.loads(Path(args.preds).read_text())
    if args.instance:
        preds = {k: v for k, v in preds.items() if k == args.instance}

    print(f"Loaded {len(preds)} predictions from {args.preds}", flush=True)

    out_path = Path(args.output)
    existing: dict[str, dict] = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            if not isinstance(existing, dict):
                existing = {}
        except Exception as e:
            print(f"Warning: could not parse existing output {out_path}: {e}", flush=True)
            existing = {}
    if existing:
        skipped = sum(1 for k in preds if k in existing)
        print(f"Resuming: {skipped} already evaluated in {out_path}, {len(preds) - skipped} remaining", flush=True)
        preds = {k: v for k, v in preds.items() if k not in existing}

    print(f"Loading {args.dataset} split={args.split} ...", flush=True)
    from datasets import load_dataset
    ds = load_dataset(args.dataset, split=args.split)
    instances = {i["instance_id"]: i for i in ds if i["instance_id"] in preds}
    print(f"Matched {len(instances)}/{len(preds)} instances in dataset\n", flush=True)

    results: dict[str, dict] = dict(existing)
    resolved_count = sum(1 for r in existing.values() if r.get("resolved"))
    done = len(existing)
    total = len(preds) + done

    out_path.parent.mkdir(parents=True, exist_ok=True)

    def save_results() -> None:
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(results, indent=2))
        tmp.replace(out_path)

    def process(iid: str) -> dict:
        pred_entry = preds[iid]
        patch = pred_entry.get("model_patch", "") or ""
        inst = instances.get(iid)
        if inst is None:
            return {"instance_id": iid, "resolved": False, "error": "not in dataset", "elapsed": 0}
        return evaluate_instance(inst, patch, args.dataset)

    if preds:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process, iid): iid for iid in preds}
            for future in as_completed(futures):
                r = future.result()
                iid = r["instance_id"]
                results[iid] = r
                done += 1
                if r.get("resolved"):
                    resolved_count += 1
                save_results()
                icon = "✓" if r.get("resolved") else "✗"
                err = f"  [{r['error']}]" if r.get("error") else ""
                f2p_str = f"  f2p={r.get('f2p_pass',0)}/{r.get('f2p_total',0)}" if "f2p_total" in r else ""
                print(
                    f"[{done:3d}/{total}] {icon} {iid}"
                    f"  ({r.get('elapsed', 0):.0f}s){f2p_str}{err}",
                    flush=True,
                )
    else:
        print("Nothing to evaluate (all preds already in output).", flush=True)

    save_results()

    rate = resolved_count / total * 100 if total else 0
    print(f"\n{'='*60}")
    print(f"Results:  {resolved_count}/{total}  ({rate:.1f}%)")
    print(f"Saved to: {out_path}")

    resolved_ids = sorted(iid for iid, r in results.items() if r.get("resolved"))
    if resolved_ids:
        print(f"\nResolved ({len(resolved_ids)}):")
        for iid in resolved_ids:
            print(f"  {iid}")

    errors = [(iid, r["error"]) for iid, r in results.items() if r.get("error")]
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for iid, err in errors[:10]:
            print(f"  {iid}: {err}")


if __name__ == "__main__":
    main()
