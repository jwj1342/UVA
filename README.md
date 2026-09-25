# UVA: Learning to Ask Without Leaking

A local student model repairs code and may consult a remote expert through one `ask_expert` action.
UVA trains the student to phrase those consultations so that private entities are described by role
and type while the technical content an expert needs is kept: the student's own consultations on
solved tasks are rewritten by an offline teacher, a rewrite is admitted only if it discloses strictly
fewer private entities and still yields a successful repair when replayed from the same execution
state, and the student is fine-tuned on the admitted questions together with ordinary repair steps.

```
uva/agent/      ask_expert channel, scheduled probe, Solo suppression, host sanitizer, replay agent
uva/privacy/    detector rules and pseudonymizer; L_P1, OCO, the resolution rule
uva/data/       candidates (teacher + checks), replay gate, master set, SFT / DPO data, task pools
uva/train/      LoRA SFT, DPO, merge for serving, one recipe per condition
uva/eval/       graders (SWE-bench Verified, SWE-smith, Terminal-Bench), pass@k / McNemar, disclosure
uva/harness/    the batch runner and Apptainer environment (override mini-swe-agent's)
configs/        collect, replay, eval (per policy and benchmark), train (per condition)
scripts/        the pipeline as shell commands
data/           task-pool id lists
```

Harness: [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) 2.2.8 from PyPI.

## Setup

```bash
pip install -r requirements.txt          # harness, data construction, scoring
pip install -r requirements-train.txt    # training (transformers 5.x)
pip install -r requirements-serve.txt    # vLLM (transformers 4.57; separate environment)
cp .env.example .env                     # endpoints and keys, read by scripts/env.sh
```

Tasks run in Apptainer sandboxes from pre-pulled images under `UVA_SIF_CACHE_DIR`:
`sweb.eval.x86_64.<instance>.sif` (Verified), `swesmith.x86_64.<repo>.sif` (SWE-smith),
`tb2/<task>.sif` (`uva/eval/terminalbench/build_tb2_images.sh`).

```bash
python -m uva.data.make_pool swesmith data/swesmith_train_ids.txt   data/pools/swesmith_train.jsonl
python -m uva.data.make_pool swesmith data/swesmith_heldout_ids.txt data/pools/swesmith_heldout.jsonl
python -m uva.data.make_pool verified data/verified_500_order.txt   data/pools/verified_500.jsonl
bash uva/eval/terminalbench/fetch_tb2_tasks.sh data/terminalbench2
bash scripts/serve_student.sh <checkpoint> [port] [max-model-len] [seed]
```

The expert (`CLOUD_*`) and the teacher (`TEACHER_*`) are OpenAI-compatible chat endpoints. A
consulting config transmits the student's question verbatim; the scripts run only with
`UVA_ACK_PUBLIC_BENCHMARK=1`.

## Pipeline

```bash
source scripts/env.sh
bash scripts/collect.sh data/pools/swesmith_train.jsonl 0:50 b0_collect        # 1. probe + score
TEACHER_MODEL=<teacher> bash scripts/build_pairs.sh data/pools/swesmith_train.jsonl b0_collect  # 2-3
python -m uva.data.pairs_master report
bash scripts/train.sh configs/train/ours.yaml <base checkpoint> 'output/runs/*_collect'         # 4
```

1. **Collect** (`configs/collect/swesmith_probe.yaml`): the student repairs the task; at step 10 and
   every 12 steps after it (at most five times) the probe has it write a consultation out of band,
   sends it verbatim and puts the reply in the trajectory. Only solved rollouts (two-sided grading)
   go on.
2. **Candidates** (`uva/data/build_candidates.py`): the teacher, given the recent history and the
   question, rewrites q- into q+. A candidate survives only if q+ has no hard secret and at most two
   detector-flagged identifiers, discloses strictly fewer provenance-verified entities than q- against
   the observations before the consultation, and is a valid plain-prose request. Identical histories
   are deduplicated, at most five candidates kept per task, rejections recorded by stage, and rewrites
   that drop a technical condition flagged for inspection.
3. **Replay** (`uva/data/replay_verify.py`): a fresh container on the instance's bug branch, the
   recorded prefix re-executed, q+ sent once with the recorded sampling seed, the student resumed with
   the remaining budget. Admitted only if the continuation resolves the task (continuations that edit
   tests are rejected). A prefix-only control gives the `load_bearing` flag; an audit subset replays
   q+ again under fresh seeds (`AUDIT_SUBSET`, `AUDIT_REPEATS`). `pairs_master report` breaks
   acceptance down by repository, step and question length.
4. **Train** (`uva/data/build_sft_mix.py`, `uva/train/sft.py`): question examples condition on the
   history h up to the consultation (middle-truncated to 9000 characters) and target
   `ask_expert "<q+>"` with the loss on q+'s tokens; solve-step examples from the same solved rollouts
   target the action, two per question. Rank-16 LoRA (alpha 32, dropout 0.05, all linear projections,
   4-bit base), 7e-5, 339 updates; merged and grafted into the base layout for serving.

## Evaluation

| policy | SWE-bench Verified | Terminal-Bench 2.0 |
|---|---|---|
| Solo | `verified_solo.yaml` | `tb2_solo.yaml` |
| Natural | `verified_natural.yaml` | `tb2_natural.yaml` |
| Privacy prompt | `verified_privacy_prompt.yaml` | `tb2_privacy_prompt.yaml` |
| Host sanitizer | `verified_host_sanitizer.yaml` | `tb2_host_sanitizer.yaml` |
| Ours | `verified_natural.yaml` + trained checkpoint | `tb2_natural.yaml` + trained checkpoint |

```bash
UVA_ACK_PUBLIC_BENCHMARK=1 bash scripts/eval_verified.sh natural verified_natural_k0   # rollouts + grading
UVA_ACK_PUBLIC_BENCHMARK=1 bash scripts/eval_tb2.sh host_sanitizer tb2_host
bash scripts/report.sh data/pools/verified_500.jsonl \
    solo=output/runs/verified_solo_k0,output/runs/verified_solo_k1,output/runs/verified_solo_k2 \
    natural=output/runs/verified_natural_k0,output/runs/verified_natural_k1,output/runs/verified_natural_k2
```

Three samples per policy (server seeds 20260716, 20260717, 20260718). `report.sh` prints pass@1 /
pass@3, solve@3 with the paired exact McNemar test and a bootstrap interval, and the disclosure table:
Ask% (rollouts with a transmitted consultation), L_P1 (inventory entities of the observations before
the consultation found in the transmitted text), OCO (source 5-grams reproduced), the detector-only
count, and the consultation accounting. The Host sanitizer is scored on its transmitted text against
the same inventory. On Terminal-Bench the task's own test script grades, and disclosure is scored
against each task's environment-definition inventory (`uva/eval/terminalbench/inventory.py`).
The Verified grader is a
local reimplementation of the official logic (`resolved` = all FAIL_TO_PASS and PASS_TO_PASS pass;
undecidable instances count as failures); `preds.json` can also be scored with the official harness.

## Ablations

| condition | recipe |
|---|---|
| Ours | `ours.yaml` |
| - solve mixture | `no_solve_mix.yaml` (`solve_per_question: 0`) |
| - replay gate | `no_replay_gate.yaml` (`replay_gate: false` on `output/pairs/all_candidates.jsonl`) |
| DPO | `dpo.yaml`; `dpo_compute_matched.yaml` (a quarter of the updates, for the extra rejected and reference passes) |
| Host-SFT | `host_sft.yaml` (targets T(q-): `TARGET=pseudonym bash scripts/build_pairs.sh ...`) |
| data scale | `datascale_{100,200,300,400}.yaml` |

All conditions share the accepted pairs, initialization, adapter and update budget
(`train.update_budget`) and are evaluated with the same eval configs.

## Fixed settings

Probe: the system prompt is swapped for a consultation-only one on the elicitation turn, one
corrective re-prompt. Expert: temperature 0.3, at most 700 tokens, fixed system prompt. Teacher:
temperature 0.3, at most 600 tokens. Student: thinking off, temperature 0.8, per-request seed during
collection (restored by the replay), server seed during evaluation; at most three consultations per
task; step budget 75 (collection, replay), 200 (Verified), 100 (Terminal-Bench). Histories are
middle-truncated to 9000 characters. Prompts: `uva/agent/prompts.py`, `uva/data/teacher.py`, the
configs.
