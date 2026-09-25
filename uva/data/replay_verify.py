#!/usr/bin/env python3
"""The replay gate.

For each candidate: a fresh container of the instance image on its bug branch, the recorded
assistant turns before the consultation re-executed (earlier consultations answered from the
record), q+ sent once with the candidate's sampling seed, the student resumed under the original
decoding with the remaining step budget. `--join` then admits a candidate only if that continuation
resolved the task. A prefix-only control that sends nothing runs alongside (the `load_bearing`
diagnostic), and an audit subset replays q+ again under fresh seeds (--audit-subset, --audit-repeats).

    python -m uva.data.replay_verify --candidates C.jsonl --config configs/replay/swesmith_replay.yaml \
        --records data/pools/swesmith_train.jsonl --runs-root output/runs --out-dir output/runs/b0_replay
    # score every output/runs/b0_replay/*_k*/preds.json with uva.eval.score_swesmith, then
    python -m uva.data.replay_verify --join --candidates C.jsonl --out-dir output/runs/b0_replay \
        --out output/pairs/b0_verified.jsonl

provenance.json records, per candidate, that exactly one consultation crossed to the expert and it
was q+, that the control sent nothing, and that the replayed prefix reproduced the recorded state
(every prefix step ran with no success-to-failure regression; output equality is a soft signal).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from uva.data.elicitation import source_trajectory
from uva.privacy import leakage as L
from uva.privacy.leakage import consultations


def _load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text())


def _records_by_id(records_path: Path) -> dict[str, dict]:
    out = {}
    for line in records_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["instance_id"]] = r
    return out


def _prefix_from_traj(traj_path: Path, call_index: int = 0) -> list[dict] | None:
    """The recorded assistant turns before the call_index-th consultation. Earlier consultations in
    the prefix are annotated (replay_ask_obs / replay_forced_obs + replay_forced_q) so the replay
    agent answers them from the record. None if the rollout has fewer consultations."""
    d = json.loads(traj_path.read_text())
    msgs = d.get("messages", [])
    cons = consultations(msgs)
    if len(cons) <= call_index:
        return None
    cut, _ = cons[call_index]
    calls = (d.get("info", {}).get("ask_expert") or {}).get("calls", [])
    prefix: list[dict] = []
    for i, m in enumerate(msgs[:cut]):
        if not (m.get("role") == "assistant" and (m.get("extra") or {}).get("actions")):
            continue
        m = copy.deepcopy(m)
        ex = m.setdefault("extra", {})
        if any((i2 == i and k2 == "spontaneous") for i2, k2 in cons):
            nxt = msgs[i + 1] if i + 1 < len(msgs) else {}
            ex["replay_ask_obs"] = nxt.get("content") if isinstance(nxt.get("content"), str) else ""
        # an EARLIER forced consultation fired after this turn: its ForcedAsk message follows the
        # observation. Only messages before the cut count; the consultation AT the cut is the one
        # this replay sends live.
        for j in range(i + 1, min(i + 3, cut)):
            mj = msgs[j]
            if mj.get("role") == "assistant":
                break
            if (mj.get("extra") or {}).get("interrupt_type") == "ForcedAsk":
                ex["replay_forced_obs"] = mj.get("content") if isinstance(mj.get("content"), str) else ""
                pos = [ci for ci, (mi, _) in enumerate(cons) if mi == j]
                if pos and pos[0] < len(calls):
                    ex["replay_forced_q"] = calls[pos[0]].get("question", "")
                break
        prefix.append(m)
    return prefix


def _answer_content(ask: dict) -> str:
    raw = ask.get("answer") or ""
    m = re.search(r"<ask_expert>(.*)</ask_expert>", raw, re.DOTALL)
    return (m.group(1) if m else raw).strip()


def _norm_output(text: str) -> str:
    """Order-insensitive normalization of a command output for cross-run comparison."""
    if text is None:
        return ""
    m = re.search(r"<output>(.*)</output>", text, re.DOTALL)
    body = m.group(1) if m else text
    return "\n".join(sorted(ln.strip() for ln in body.splitlines() if ln.strip()))


def _prefix_obs(traj_path: Path, n: int) -> list:
    """(return_code, normalized output) after each of the first n assistant-with-actions turns."""
    d = json.loads(traj_path.read_text())
    msgs = d.get("messages", [])
    out = []
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant" and (m.get("extra") or {}).get("actions"):
            obs = msgs[i + 1] if i + 1 < len(msgs) else {}
            rc = (obs.get("extra") or {}).get("returncode")
            content = obs.get("content") if isinstance(obs.get("content"), str) else ""
            out.append((rc, _norm_output(content)))
            if len(out) >= n:
                break
    return out


def _restoration_ok(source_obs: list, replay_traj: Path, n: int) -> tuple[bool, str, bool, str]:
    """(hard_ok, hard_why, output_match, output_why): hard = all n prefix steps ran with no
    success-to-failure return-code regression (gates acceptance); soft = normalized outputs match."""
    if not replay_traj.exists():
        return False, "replay trajectory missing", False, "replay trajectory missing"
    replay_obs = _prefix_obs(replay_traj, n)
    if len(replay_obs) < n:
        why = f"replay executed {len(replay_obs)}/{n} prefix steps"
        return False, why, False, why
    output_match, output_why = True, ""
    for i in range(n):
        s_rc, s_out = source_obs[i]
        r_rc, r_out = replay_obs[i]
        if s_rc == 0 and r_rc != 0:
            return False, f"prefix step {i}: source rc=0 but replay rc={r_rc}", output_match, output_why
        if output_match and s_out != r_out:
            output_match, output_why = False, f"prefix step {i}: output diverged"
    return True, "", output_match, output_why


def _model_config(config: dict, seed=None) -> dict:
    mc = copy.deepcopy(config.get("model", {}))
    base = os.environ.get("STUDENT_API_BASE") or os.environ.get("HOSTED_VLLM_API_BASE")
    if base:
        mc.setdefault("model_kwargs", {}).setdefault("api_base", base)
    if seed is not None:
        mc.setdefault("model_kwargs", {})["seed"] = int(seed)
    return mc


def _run_arm(model, env_factory, config, instance, prefix, q_send, out_path: Path):
    """One replay rollout. q_send='' is the prefix-only control. Returns (info, asks)."""
    from uva.agent.ask_probe import ReplayAskProbeAgent
    env = env_factory()
    try:
        agent_cfg = copy.deepcopy(config.get("agent", {}))
        agent_cfg.pop("agent_class", None)
        agent_cfg["force_ask_step"] = 0
        agent_cfg["force_ask_reask_every"] = 0
        agent_cfg["max_ask_calls"] = 1            # exactly one live consultation may be sent
        agent_cfg["replay_question"] = q_send
        agent_cfg["suppress_ask"] = (q_send == "")  # the control sends nothing
        agent_cfg["output_path"] = out_path
        agent = ReplayAskProbeAgent(model, env, **agent_cfg)
        agent.replay_prefix = copy.deepcopy(prefix)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        info = agent.run(instance["problem_statement"])
        return info, list(agent.asks)
    finally:
        try:
            env.cleanup()
        except Exception:  # noqa: BLE001
            pass


def _write_pred(path: Path, iid: str, model_name: str, patch: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[iid] = {"instance_id": iid, "model_name_or_path": model_name, "model_patch": patch}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def run_rollouts(args) -> None:
    from minisweagent.models import get_model
    from uva.harness.run_swebench import get_sb_environment

    config = _load_yaml(Path(args.config))
    records = _records_by_id(Path(args.records))
    candidates = [json.loads(l) for l in Path(args.candidates).read_text().splitlines() if l.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for arm_dir in out_dir.glob("*_k[0-9]*"):
        for stale in ("preds.json", "eval_results.json"):
            (arm_dir / stale).unlink(missing_ok=True)
    prov: dict[str, dict] = {}
    skips: dict[str, str] = {}
    seen: set[tuple[int, str]] = set()  # one candidate per (call_index, instance)

    def _flush() -> None:
        for name, obj in (("provenance.json", prov), ("skips.json", skips)):
            tmp = out_dir / (name + ".tmp")
            tmp.write_text(json.dumps(obj, indent=2))
            tmp.replace(out_dir / name)

    n_ok = n_skip = 0
    for cand in candidates:
        iid = cand["instance_id"]
        k = int(cand.get("call_index", 0))
        cid = f"{iid}__{cand.get('traj_sha16', '')}__{k}"

        def _skip(reason: str) -> None:
            print(f"[replay] skip {cid}: {reason}", file=sys.stderr)
            skips[cid] = reason
            _flush()

        if (k, iid) in seen:
            _skip("duplicate_instance_id"); n_skip += 1; continue
        if iid not in records:
            _skip("not_in_records"); n_skip += 1; continue
        src = source_trajectory(args.runs_root, cand["run"], iid)
        if not src.exists():
            _skip("source_traj_missing"); n_skip += 1; continue
        if hashlib.sha256(src.read_bytes()).hexdigest()[:16] != cand.get("traj_sha16"):
            _skip("source_traj_hash_mismatch"); n_skip += 1; continue
        prefix = _prefix_from_traj(src, k)
        if not prefix:
            _skip("no_prefix_for_call_index"); n_skip += 1; continue
        seen.add((k, iid))

        instance = records[iid]
        source_obs = _prefix_obs(src, len(prefix))

        def factory():
            return get_sb_environment(config, instance)

        arm_q, arm_n = f"qplus_k{k}", f"noask_k{k}"
        qplus_traj = out_dir / arm_q / iid / f"{iid}.traj.json"
        noask_traj = out_dir / arm_n / iid / f"{iid}.traj.json"
        seed = cand.get("seed")
        audited = n_ok < args.audit_subset
        try:
            model = get_model(config=_model_config(config, seed))
            info_q, asks_q = _run_arm(model, factory, config, instance, prefix, cand["chosen"], qplus_traj)
            info_n, asks_n = _run_arm(model, factory, config, instance, prefix, "", noask_traj)
            repeats = []
            for r_i in range(args.audit_repeats if audited else 0):
                rep_seed = (int(seed) if seed is not None else 0) + 1000 + r_i
                rep_traj = out_dir / f"{arm_q}_rep{r_i}" / iid / f"{iid}.traj.json"
                info_r, asks_r = _run_arm(get_model(config=_model_config(config, rep_seed)), factory, config,
                                          instance, prefix, cand["chosen"], rep_traj)
                raw_r = info_r.get("submission") or ""
                if raw_r.strip():
                    _write_pred(out_dir / f"{arm_q}_rep{r_i}" / "preds.json", iid,
                                config.get("model", {}).get("model_name", ""), raw_r)
                repeats.append({"arm": f"{arm_q}_rep{r_i}", "seed": rep_seed, "submitted": bool(raw_r.strip()),
                                "sent": sum(1 for a in asks_r if a.get("outcome") == "answered")})
        except Exception as e:  # noqa: BLE001 -- one bad candidate must not stop the batch
            print(f"[replay] {cid}: run failed {type(e).__name__}: {e}", file=sys.stderr)
            _skip(f"run_failed:{type(e).__name__}")
            n_skip += 1
            continue

        q_restore, q_why, q_omatch, q_owhy = _restoration_ok(source_obs, qplus_traj, len(prefix))
        n_restore, n_why, n_omatch, n_owhy = _restoration_ok(source_obs, noask_traj, len(prefix))
        if not q_restore or not n_restore:
            print(f"[replay] {cid}: prefix NOT restored (qplus: {q_why or 'ok'} / noask: {n_why or 'ok'})",
                  file=sys.stderr)

        sent = [a for a in asks_q if a.get("outcome") == "answered"]
        consult_ok = (len(sent) == 1 and bool(sent[0].get("forced"))
                      and sent[0].get("question", "").strip() == cand["chosen"].strip()
                      and bool(_answer_content(sent[0])))
        # raw submission: `git apply` rejects a diff that lost its trailing newline
        raw_q = info_q.get("submission") or ""
        raw_n = info_n.get("submission") or ""
        sub_q, sub_n = raw_q.strip(), raw_n.strip()
        for arm, raw, sub in ((arm_q, raw_q, sub_q), (arm_n, raw_n, sub_n)):
            if sub:
                _write_pred(out_dir / arm / "preds.json", iid, config.get("model", {}).get("model_name", ""), raw)
        prov[cid] = {"instance_id": iid, "call_index": k, "consult_ok": bool(consult_ok),
                     "prefix_replayed_consultations": sum(1 for a in asks_q if a.get("outcome") == "replayed"),
                     "qplus_submitted": bool(sub_q), "noask_submitted": bool(sub_n),
                     "qplus_sent": len(sent),
                     "noask_sent": sum(1 for a in asks_n if a.get("outcome") == "answered"),
                     "qplus_restore_ok": bool(q_restore), "noask_restore_ok": bool(n_restore),
                     "qplus_output_match": bool(q_omatch), "noask_output_match": bool(n_omatch),
                     "prefix_steps": len(prefix), "seed": seed, "audit_repeats": repeats,
                     "qplus_exit": info_q.get("exit_status"), "noask_exit": info_n.get("exit_status")}
        n_ok += 1
        _flush()
        print(f"[replay] {cid}: consult_ok={prov[cid]['consult_ok']} restore(q/n)="
              f"{int(q_restore)}/{int(n_restore)} q+patch={'y' if sub_q else 'n'} "
              f"noask_patch={'y' if sub_n else 'n'}", file=sys.stderr)

    _flush()
    print(f"[replay] ran {n_ok} candidates, skipped {n_skip}; provenance -> {out_dir}/provenance.json",
          file=sys.stderr)


def join(args) -> None:
    """Label every candidate from the scored replay arms:
      accepted                     q+ resolved, the control did not (load-bearing)
      uninformative_prefix_solved  q+ resolved and so did the control (admitted)
      rejected_replay              q+ answered but the continuation did not resolve the task
      rejected_modified_tests      the continuation's patch edits test files
      rejected_no_consult          q+ was not actually sent and answered
      replay_error                 no verdict: never ran, state not restored, control consulted,
                                   or a submitted arm graded indeterminately"""
    out_dir = Path(args.out_dir)
    prov = json.loads((out_dir / "provenance.json").read_text()) if (out_dir / "provenance.json").exists() else {}

    def _indeterminate(v: dict) -> bool:
        if v.get("harness_error"):
            return True
        if v.get("f2p_resolved") and not v.get("resolved") and not v.get("p2p_skipped") \
                and v.get("p2p_pass", 0) < v.get("p2p_total", 0):
            return True  # a PASS_TO_PASS shortfall: regression or infrastructure, undecidable here
        brm = v.get("baseline_f2p_run_missing")
        return brm is not None and brm > 0

    def _scored(arm):
        ev = out_dir / arm / "eval_results.json"
        data = json.loads(ev.read_text()) if ev.exists() else {}
        resolved = {k for k, v in data.items() if L.strict_resolved(v, "smith")}
        clean = {k for k, v in data.items() if isinstance(v, dict) and not _indeterminate(v)}
        return resolved, clean

    def _touches_tests(arm, iid) -> bool:
        """A replay whose submitted patch edits test files is rejected rather than graded."""
        preds = out_dir / arm / "preds.json"
        patch = (json.loads(preds.read_text()).get(iid, {}).get("model_patch", "") if preds.exists() else "")
        for line in patch.splitlines():
            if line.startswith("diff --git "):
                path = line.split(" b/", 1)[-1].strip().lower()
                if re.search(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*\.py$|_test\.py$|conftest\.py$", path):
                    return True
        return False

    cands = [json.loads(l) for l in Path(args.candidates).read_text().splitlines() if l.strip()]
    cache: dict[str, tuple[set, set]] = {}
    from collections import Counter
    status_count: Counter = Counter()
    audit = {"audited": 0, "repeats": 0, "repeats_resolved": 0, "all_repeats_resolved": 0}
    with Path(args.out).open("w") as fh:
        for c in cands:
            iid = c["instance_id"]
            k = int(c.get("call_index", 0))
            cid = f"{iid}__{c.get('traj_sha16', '')}__{k}"
            p = prov.get(cid, {})
            for arm in (f"qplus_k{k}", f"noask_k{k}"):
                if arm not in cache:
                    cache[arm] = _scored(arm)
            res_q, scored_q = cache[f"qplus_k{k}"]
            res_n, scored_n = cache[f"noask_k{k}"]
            q_res = bool(p.get("qplus_submitted")) and iid in res_q
            n_res = bool(p.get("noask_submitted")) and iid in res_n
            consult_ok = bool(p.get("consult_ok"))
            q_evidence_ok = (not p.get("qplus_submitted")) or (iid in scored_q)
            n_evidence_ok = (not p.get("noask_submitted")) or (iid in scored_n)
            restore_ok = p.get("qplus_restore_ok", True) and p.get("noask_restore_ok", True)
            if not p:
                status = "replay_error"
            elif p.get("noask_sent", 0) > 0:
                status = "replay_error"
            elif not restore_ok:
                status = "replay_error"
            elif not consult_ok:
                status = "rejected_no_consult"
            elif p.get("qplus_submitted") and _touches_tests(f"qplus_k{k}", iid):
                status = "rejected_modified_tests"
            elif not (q_evidence_ok and n_evidence_ok):
                status = "replay_error"
            elif not q_res:
                status = "rejected_replay"                # S(tau+) = 0: the gate rejects
            elif n_res:
                status = "uninformative_prefix_solved"    # q+ resolved, but so did the control
            else:
                status = "accepted"                       # q+ resolved and the control did not
            status_count[status] += 1
            reps = p.get("audit_repeats") or []
            rep_res = []
            for rep in reps:
                if rep["arm"] not in cache:
                    cache[rep["arm"]] = _scored(rep["arm"])
                rep_res.append(bool(rep.get("submitted")) and iid in cache[rep["arm"]][0])
            if reps and status in ("accepted", "uninformative_prefix_solved"):
                audit["audited"] += 1
                audit["repeats"] += len(rep_res)
                audit["repeats_resolved"] += sum(rep_res)
                audit["all_repeats_resolved"] += int(all(rep_res))
            c.update({"replay_verified": status == "accepted", "status": status,
                      "qplus_resolved": q_res, "noask_resolved": n_res, "consult_ok": consult_ok,
                      "repeat_resolved": rep_res})
            fh.write(json.dumps(c) + "\n")
    print(f"[join] {dict(status_count)} -> {args.out}", file=sys.stderr)
    if audit["audited"]:
        print(f"[join] repeated-replay audit: {audit['audited']} admitted candidates x repeats; "
              f"{audit['repeats_resolved']}/{audit['repeats']} repeats resolved, "
              f"{audit['all_repeats_resolved']}/{audit['audited']} reproduce on every repeat", file=sys.stderr)
        Path(args.out).with_suffix(".audit.json").write_text(json.dumps(audit, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", help="replay config (same prompts/decoding as the collection)")
    ap.add_argument("--records", help="task records jsonl (instance_id, image_name, problem_statement, ...)")
    ap.add_argument("--runs-root", default="output/runs", help="where the collection runs live")
    ap.add_argument("--out", help="--join: verified candidates jsonl")
    ap.add_argument("--join", action="store_true", help="join scored results instead of running")
    ap.add_argument("--audit-subset", type=int, default=0, help="first K candidates get repeated q+ replays")
    ap.add_argument("--audit-repeats", type=int, default=2, help="extra q+ replays per audited candidate")
    args = ap.parse_args()
    if args.join:
        if not args.out:
            ap.error("--join needs --out")
        join(args)
    else:
        if not (args.config and args.records):
            ap.error("rollouts need --config and --records")
        run_rollouts(args)


if __name__ == "__main__":
    main()
