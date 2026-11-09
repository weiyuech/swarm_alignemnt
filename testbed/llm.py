"""Model clients. Keys are read from macOS Keychain into memory only.

Providers:
  mock     - deterministic scripted policy, for tests (no network)
  adaption - Adaption hosted inference, POST /api/v1/responses
  openai   - any OpenAI-compatible /chat/completions endpoint (used for Gemini)

Each call returns an LLMResult with text, token usage, cost and latency. The
key never leaves this module: it is not logged, not put in prompts and not
passed to the grader subprocess.
"""

import getpass
import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

KEYCHAIN = {
    "adaption": "apart-hackathon-adaption-api-key",
    "gemini": "apart-hackathon-gemini-api-key",
}

ENDPOINTS = {
    "adaption": "https://api.prod.adaptionlabs.ai/api/v1/responses",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
}

# USD (Adaption credits) per million tokens: input, output.
# Source: docs.adaptionlabs.ai/inference-stacks/models (checked Nov 7, 2026).
PRICES = {
    "deepseek-v4-flash": (0.04444, 0.13332),
    "glm-53-flash": (0.15, 0.50),
    "gemma-4-31b-it": (0.09, 0.34),
    "gemma-4-26b-a4b-it": (0.0765, 0.255),
    "qwen3.8-27b": (0.039, 1.25),
    "qwen3.5-397b": (0.55, 3.50),
    "glm-5.2": (1.40, 4.40),
}


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost: float = 0.0
    latency_s: float = 0.0
    status: str = "ok"
    error: str = ""
    raw_meta: Dict = field(default_factory=dict)


def keychain_secret(service):
    out = subprocess.run(
        ["/usr/bin/security", "find-generic-password", "-a", getpass.getuser(), "-s", service, "-w"],
        capture_output=True, text=True, check=True)
    secret = out.stdout.strip()
    if not secret:
        raise RuntimeError("empty credential in Keychain item " + service)
    return secret


class BaseClient:
    def __init__(self, model, max_output_tokens=2048, temperature=1.0, timeout_s=180, max_retries=3):
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def complete(self, system, user, agent_id="", turn=0):
        raise NotImplementedError

    def _price(self, inp, out):
        p_in, p_out = PRICES.get(self.model, (0.0, 0.0))
        return inp * p_in / 1e6 + out * p_out / 1e6

    def _post(self, url, key, body):
        data = json.dumps(body).encode()
        last_err = ""
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(url, data=data, method="POST", headers={
                "Authorization": "Bearer " + key, "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    return json.load(resp), ""
            except urllib.error.HTTPError as ex:
                last_err = "HTTP %d" % ex.code
                if ex.code not in (408, 409, 429, 500, 502, 503, 504):
                    break
            except (urllib.error.URLError, TimeoutError, OSError) as ex:
                last_err = type(ex).__name__
            time.sleep(min(2 ** attempt * 2, 30))
        return None, last_err


class MockClient(BaseClient):
    """policy(agent_id, turn, system, user) -> JSON text."""

    def __init__(self, policy: Callable, **kw):
        super().__init__("mock", **kw)
        self.policy = policy

    def complete(self, system, user, agent_id="", turn=0):
        return LLMResult(text=self.policy(agent_id, turn, system, user))


class AdaptionClient(BaseClient):
    def __init__(self, model, reasoning_effort=None, deadline_s=None, **kw):
        super().__init__(model, **kw)
        if deadline_s:
            self.deadline_s = deadline_s
        self._key = keychain_secret(KEYCHAIN["adaption"])
        # On deepseek-v4-flash, effort "none" disables hidden reasoning; "low" and
        # "medium" had no reliable effect in our probes (Nov 8).
        self.reasoning_effort = reasoning_effort

    # Non-streamed requests are dropped by the gateway after ~60 s, and reasoning
    # models routinely think longer than that, so responses are streamed (SSE).
    # `deadline_s` caps wall-clock time per call; a call that hits it, or hits
    # max_output_tokens while still reasoning, returns status "incomplete" and
    # the agent simply does not act that turn (same rule in every condition).
    deadline_s = 240

    def complete(self, system, user, agent_id="", turn=0):
        body = {
            "model": self.model,
            "input": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "tools": [], "tool_choice": "none",
            "store": False, "agentic_memory": False, "stream": True,
        }
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        t0 = time.time()
        final, deltas, err = None, [], ""
        for attempt in range(self.max_retries + 1):
            got_data = False
            req = urllib.request.Request(ENDPOINTS["adaption"], data=json.dumps(body).encode(), method="POST",
                                         headers={"Authorization": "Bearer " + self._key,
                                                  "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    for raw in resp:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        got_data = True
                        payload = line[5:].strip()
                        if not payload or payload == "[DONE]":
                            continue
                        try:
                            ev = json.loads(payload)
                        except ValueError:
                            continue
                        typ = ev.get("type", "")
                        if typ == "response.output_text.delta":
                            deltas.append(ev.get("delta", ""))
                        elif typ in ("response.completed", "response.incomplete", "response.failed"):
                            final = ev.get("response") or {}
                        if time.time() - t0 > self.deadline_s:
                            err = "deadline"
                            break
                err = err or ("" if final is not None else "stream ended without a final event")
                break
            except urllib.error.HTTPError as ex:
                err = "HTTP %d" % ex.code
                if ex.code not in (408, 409, 429, 500, 502, 503, 504):
                    break
            except (urllib.error.URLError, TimeoutError, OSError) as ex:
                err = type(ex).__name__
            if got_data:
                break                      # never resend a request that already produced output
            time.sleep(min(2 ** attempt * 2, 30))
        dt = time.time() - t0
        j = final or {}
        text = j.get("output_text") or "".join(deltas)
        reasoning_text = ""
        for item in j.get("output") or []:
            if item.get("type") == "reasoning":
                for c in item.get("content") or []:
                    if isinstance(c, dict) and c.get("text"):
                        reasoning_text += c["text"]
                for s in item.get("summary") or []:
                    if isinstance(s, dict) and s.get("text"):
                        reasoning_text += s["text"]
            elif not j.get("output_text"):
                for c in item.get("content") or []:
                    if isinstance(c, dict) and c.get("type") in ("output_text", "text") and c.get("text"):
                        text += c["text"]
        u = j.get("usage") or {}
        inp, out = u.get("input_tokens", 0) or 0, u.get("output_tokens", 0) or 0
        reasoning = (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        if final is None:
            status = "error"
        elif err == "deadline":
            status = "incomplete"
        else:
            status = j.get("status", "completed")
        cost = self._price(inp, out)
        if final is None or not u:
            # No usage reported (dropped stream or deadline): charge the worst
            # case so the budget guard never undercounts.
            cost = self._price((len(system) + len(user)) // 3, self.max_output_tokens)
        return LLMResult(text=text, input_tokens=inp, output_tokens=out, reasoning_tokens=reasoning,
                         cost=cost, latency_s=dt, status=status, error=err,
                         raw_meta={"id": j.get("id"), "model": j.get("model"),
                                   "incomplete_reason": (j.get("incomplete_details") or {}).get("reason"),
                                   "reasoning_text": reasoning_text})


class OpenAICompatClient(BaseClient):
    def __init__(self, model, provider="gemini", prices=None, **kw):
        super().__init__(model, **kw)
        self.url = ENDPOINTS[provider]
        self._key = keychain_secret(KEYCHAIN[provider])
        if prices:
            PRICES[model] = prices

    def complete(self, system, user, agent_id="", turn=0):
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }
        t0 = time.time()
        j, err = self._post(self.url, self._key, body)
        dt = time.time() - t0
        if j is None:
            return LLMResult(text="", latency_s=dt, status="error", error=err)
        try:
            text = j["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            text = ""
        u = j.get("usage") or {}
        inp, out = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        reasoning = (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        return LLMResult(text=text, input_tokens=inp, output_tokens=out, reasoning_tokens=reasoning,
                         cost=self._price(inp, out), latency_s=dt,
                         raw_meta={"id": j.get("id"), "model": j.get("model")})


def make_client(provider, model, **kw):
    if provider == "adaption":
        return AdaptionClient(model, **kw)
    if provider == "gemini":
        return OpenAICompatClient(model, provider="gemini", **kw)
    raise ValueError("unknown provider " + provider)
