"""The consultation channel: mini-swe-agent's DefaultAgent plus a model-invoked `ask_expert` action.

A command block that starts with `ask_expert "..."` is not executed in the container: the question
is sent to the remote expert and the reply comes back as that command's output. One config class
covers every policy:

    cloud_reply      send the question and inject the real answer (all consulting policies)
    suppress_ask     refuse every consultation but record the attempt (Solo)
    host_sanitize    pseudonymize the outbound question with T, reverse-map the reply (Host sanitizer)
    force_ask_step / force_ask_reask_every
                     the scheduled probe of data collection: at these steps the student writes a
                     consultation out of band; the probe instruction never enters the trajectory
    replay_question  ReplayAskProbeAgent only: the q+ to send when a replayed prefix is exhausted
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from pathlib import Path

from minisweagent.agents.default import AgentConfig, DefaultAgent

from uva.agent.ask_syntax import ASK, looks_truncated, parse_ask
from uva.agent.prompts import (CLOUD_SYSTEM, CONSULT_SYSTEM, EXHAUSTED, FORCE_INSTRUCTION, STUB,
                               SUPPRESSED)

_FENCE = "```"
_SHELL_START = re.compile(
    r"^\s*(cd|ls|cat|grep|rg|python3?|pytest|pip|git|sed|awk|echo|printf|rm|mv|cp|find|"
    r"chmod|chown|sudo|bash|sh|make|export|touch|mkdir|head|tail|diff|apply|patch)\b")


class AskProbeConfig(AgentConfig):
    max_ask_calls: int = 0          # consultation budget per task; 0 = unlimited (not suppression)
    force_ask_step: int = 0         # scheduled probe: first forced consultation at this step; 0 = off
    force_ask_reask_every: int = 0  # then every N steps; 0 = once
    suppress_ask: bool = False      # refuse every consultation, record the attempt
    cloud_reply: bool = False       # send to the expert (else a stub reply)
    cloud_model: str = "deepseek/deepseek-v3.2"
    cloud_timeout: int = 180
    host_sanitize: bool = False     # pseudonymize outbound text, reverse-map the reply
    replay_question: str = ""       # ReplayAskProbeAgent: the q+ to send


class AskProbeAgent(DefaultAgent):
    def __init__(self, model, env, *, config_class: type = AskProbeConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.asks: list[dict] = []
        self.oob_calls: int = 0
        self.oob_cost: float = 0.0
        self._forced_steps: list[int] = []
        self._last_call: dict = {}

    @staticmethod
    def _parse_ask(command: str) -> str | None:
        return parse_ask(command)

    # ------------------------------------------------------------------ the expert
    def _consult(self, question: str) -> tuple[str, str]:
        """Send one question to the expert. Returns (observation, outcome); token usage and latency
        of the call are kept in self._last_call for the consultation accounting."""
        self._last_call = {}
        base = os.environ.get("CLOUD_API_BASE", "").rstrip("/")
        key = os.environ.get("CLOUD_API_KEY", "")
        key_file = os.environ.get("CLOUD_API_KEY_FILE", "")
        if not key and key_file and Path(key_file).exists():
            key = Path(key_file).read_text().strip().splitlines()[-1].strip()
        if not base or not key:
            return ("<ask_expert>\nHARNESS ERROR: no cloud endpoint configured; this question "
                    "was not sent.\n</ask_expert>", "misconfigured")
        body = json.dumps({
            "model": self.config.cloud_model,
            "messages": [{"role": "system", "content": CLOUD_SYSTEM},
                         {"role": "user", "content": question}],
            "temperature": 0.3, "max_tokens": 700,
        }).encode()
        req = urllib.request.Request(f"{base}/chat/completions", data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.config.cloud_timeout) as r:
                data = json.load(r)
            answer = data["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001 -- recorded in the trajectory, never fatal
            self._last_call = {"latency_s": round(time.monotonic() - t0, 3)}
            return (f"<ask_expert>\nThe question could not be delivered ({type(e).__name__}). "
                    f"Continue working locally.\n</ask_expert>", f"failed:{type(e).__name__}")
        usage = data.get("usage") or {}
        self._last_call = {"latency_s": round(time.monotonic() - t0, 3),
                           "expert_input_tokens": usage.get("prompt_tokens"),
                           "expert_output_tokens": usage.get("completion_tokens")}
        if not answer:
            return ("<ask_expert>\nThe expert returned an empty reply. Continue working locally.\n</ask_expert>",
                    "empty_reply")
        return (f"<ask_expert>\n{answer}\n</ask_expert>", "answered")

    def _consult_sanitized(self, question: str) -> tuple[str, str, str]:
        """Host sanitizer: T on the outbound text, placeholders restored in the reply.
        Returns (observation, outcome, sent_question)."""
        from uva.privacy.detector import Pseudonymizer
        psx = Pseudonymizer()
        sent = psx.transform(question, include_p1=True)
        mapping = dict(psx.mapping)
        obs, outcome = self._consult(sent)
        if outcome == "answered" and mapping:
            for ph in sorted(mapping, key=len, reverse=True):  # IDENT_10 before IDENT_1
                obs = obs.replace(ph, mapping[ph])
        return obs, outcome, sent

    # ------------------------------------------------------------------ the scheduled probe
    def _elicit_question(self) -> tuple[str | None, str]:
        """Have the model write a consultation on a throwaway copy of its history. Returns
        (question, note); one corrective re-prompt when the reply is not plain prose."""
        has_system = bool(self.messages) and self.messages[0].get("role") == "system"
        body = list(self.messages[1:]) if has_system else list(self.messages)
        history = ([{"role": "system", "content": CONSULT_SYSTEM}] + body
                   + [{"role": "user", "content": FORCE_INSTRUCTION}])
        note = "ok"
        for _ in range(2):
            text, finish_reason = self._query_prose(history)
            if text is None and finish_reason is None:
                return None, "elicit_failed"
            q = self._clean_question(text, finish_reason)
            if q is not None:
                return q, note
            history = history + [
                {"role": "assistant", "content": text or ""},
                {"role": "user", "content": (
                    "That reply was not a plain-text question (it was a command, a code fence, "
                    "or was cut off). Reply with ONE short plain-text question, no backticks, "
                    "no code, no command.")}]
            note = "reprompted"
        return None, "elicit_dirty"

    def _clean_question(self, text: str | None, finish_reason: str | None) -> str | None:
        """A usable plain-text question, or None (empty, cut off, fenced, or a shell command)."""
        if text is None:
            return None
        t = text.strip()
        if not t or finish_reason == "length" or _FENCE in t:
            return None
        if ASK.match(t):
            q = (parse_ask(t) or "").strip()
            return None if (not q or _FENCE in q or _SHELL_START.match(q)) else q
        return None if _SHELL_START.match(t) else t

    def _query_prose(self, history: list) -> tuple[str | None, str | None]:
        """One out-of-band model call for prose (bypasses the action parser). Counted in cost,
        not as a trajectory step."""
        try:
            prepared = self.model._prepare_messages_for_api(history)
        except Exception:  # noqa: BLE001
            return None, None
        response = None
        for _ in range(3):
            try:
                response = self.model._query(prepared)
                break
            except Exception:  # noqa: BLE001
                response = None
        if response is None:
            return None, None
        self.oob_calls += 1
        try:
            cost = float(self.model._calculate_cost(response).get("cost", 0.0))
            self.oob_cost += cost
            self.cost += cost
        except Exception:  # noqa: BLE001
            pass
        try:
            choice = response.choices[0]
            return (choice.message.content or "").strip(), getattr(choice, "finish_reason", None)
        except Exception:  # noqa: BLE001
            return None, None

    def _due_for_forced_ask(self) -> bool:
        t = self.config.force_ask_step
        if t <= 0:
            return False
        if not self._forced_steps:
            return self.n_calls >= t
        every = self.config.force_ask_reask_every
        return bool(every) and self.n_calls - self._forced_steps[-1] >= every

    # ------------------------------------------------------------------ recording
    def _record_and_answer(self, question: str, *, forced: bool, elicit_note: str = "") -> str:
        cap = self.config.max_ask_calls
        sent_question = question
        if self.config.suppress_ask:
            outcome, obs = "suppressed", SUPPRESSED
        elif cap and len([a for a in self.asks
                          if a.get("outcome") not in ("over_budget", "replayed")]) >= cap:
            outcome, obs = "over_budget", EXHAUSTED
        elif self.config.cloud_reply and self.config.host_sanitize:
            obs, outcome, sent_question = self._consult_sanitized(question)
        elif self.config.cloud_reply:
            obs, outcome = self._consult(question)
        else:
            obs, outcome = STUB, "recorded_only"
        rec = {"step": self.n_calls, "question": question, "outcome": outcome,
               "answer": obs if outcome == "answered" else "",
               "truncated": looks_truncated(question), "elicit_note": elicit_note, "forced": forced,
               **self._last_call}
        self._last_call = {}
        if self.config.host_sanitize:
            rec["sent_question"] = sent_question  # disclosure is measured on the transmitted text
        self.asks.append(rec)
        return obs

    def execute_actions(self, message: dict) -> list[dict]:
        outputs = []
        for action in message.get("extra", {}).get("actions", []):
            question = self._parse_ask(action.get("command", ""))
            if question is None:
                outputs.append(self.env.execute(action))
                continue
            outputs.append({"output": self._record_and_answer(question, forced=False),
                            "returncode": 0, "exception_info": ""})
        observations = self.add_messages(
            *self.model.format_observation_messages(message, outputs, self.get_template_vars()))
        if self._due_for_forced_ask():  # after the step's observations, like a spontaneous ask
            self._forced_steps.append(self.n_calls)
            q, note = self._elicit_question()
            if q is None:
                self.asks.append({"step": self.n_calls, "question": "", "outcome": note, "answer": "",
                                  "truncated": False, "elicit_note": note, "forced": True})
            else:
                obs = self._record_and_answer(q, forced=True, elicit_note=note)
                self.add_messages({"role": "user", "content": obs,
                                   "extra": {"interrupt_type": "ForcedAsk", "step": self.n_calls,
                                             "elicit_note": note}})
        return observations

    def _arm_label(self) -> str:
        if self.config.force_ask_step > 0:
            return "forced"
        return "suppressed" if self.config.suppress_ask else "free"

    def serialize(self, *extra_dicts) -> dict:
        return super().serialize(
            {"info": {
                "ask_expert": {"n": len(self.asks), "calls": self.asks,
                               "oob_calls": self.oob_calls, "oob_cost": round(self.oob_cost, 6)},
                "decision_point": {"force_ask_step": self.config.force_ask_step,
                                   "force_ask_reask_every": self.config.force_ask_reask_every,
                                   "suppress_ask": self.config.suppress_ask,
                                   "host_sanitize": self.config.host_sanitize,
                                   "forced_steps": self._forced_steps, "arm": self._arm_label()},
            }}, *extra_dicts)


class ReplayAskProbeAgent(AskProbeAgent):
    """Replay for the acceptance gate: re-execute a recorded prefix of assistant turns (rebuilding
    the container state), send `replay_question` exactly when the prefix is exhausted, and continue
    autonomously. Earlier consultations inside the prefix are answered from the record (their
    `replay_ask_obs` / `replay_forced_obs` annotations), never re-sent. With an empty
    `replay_question` and `suppress_ask` it is the prefix-only control."""

    def __init__(self, model, env, **kwargs):
        super().__init__(model, env, **kwargs)
        self.replay_prefix: list[dict] = []
        self._consulted: bool = False

    def query(self) -> dict:
        if self.replay_prefix:  # a replayed turn spends one call, so the remaining budget is preserved
            msg = self.replay_prefix.pop(0)
            self.n_calls += 1
            self.add_messages(msg)
            return msg
        return super().query()

    def execute_actions(self, message: dict) -> list[dict]:
        ex = message.get("extra") or {}
        if "replay_ask_obs" not in ex and "replay_forced_obs" not in ex:
            return super().execute_actions(message)
        outputs = []
        for action in ex.get("actions", []):
            question = self._parse_ask(action.get("command", ""))
            if question is None or "replay_ask_obs" not in ex:
                outputs.append(self.env.execute(action))
                continue
            outputs.append({"output": ex["replay_ask_obs"], "returncode": 0, "exception_info": ""})
            self.asks.append({"step": self.n_calls, "question": question, "outcome": "replayed",
                              "answer": "", "truncated": False, "elicit_note": "replayed_from_record",
                              "forced": False})
        observations = self.add_messages(
            *self.model.format_observation_messages(message, outputs, self.get_template_vars()))
        if "replay_forced_obs" in ex:
            self.asks.append({"step": self.n_calls, "question": ex.get("replay_forced_q", ""),
                              "outcome": "replayed", "answer": "", "truncated": False,
                              "elicit_note": "replayed_from_record", "forced": True})
            self.add_messages({"role": "user", "content": ex["replay_forced_obs"],
                               "extra": {"interrupt_type": "ForcedAsk", "step": self.n_calls,
                                         "elicit_note": "replayed_from_record"}})
        return observations

    def _due_for_forced_ask(self) -> bool:
        if self._consulted or self.replay_prefix:
            return False
        self._consulted = True
        return True

    def _elicit_question(self) -> tuple[str | None, str]:
        q = (self.config.replay_question or "").strip()
        return (q, "replay") if q else (None, "replay_empty")
