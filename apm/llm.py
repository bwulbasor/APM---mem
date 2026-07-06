"""LLM transducer client (Phase 3/4: PARSE and VERBALIZE — the only two LLM jobs).

OpenAI-compatible chat endpoint, zero dependencies. Config from .env / environment:
  APM_LLM_BASE_URL, APM_LLM_API_KEY
  APM_LLM_MODEL                default model for all roles
  APM_LLM_MODEL_<ROLE>         per-role override (EXTRACT, PARSE, VERBALIZE,
                               VETO, JUDGE) — transducer jobs are shallow
                               structured tasks; fast instruct models beat
                               reasoning models on them
  APM_LLM_CACHE                sqlite path ("" disables). Content-keyed
                               memoization: hash(model+system+prompt+params) ->
                               the model's own output. Stores nothing derived
                               from gold answers; makes runs reproducible so
                               code changes become exact ablations.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sqlite3
import time
import urllib.error
import urllib.request


def _load_env():
    env_file = pathlib.Path(__file__).resolve().parents[1] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


_load_env()

BASE_URL = os.environ.get("APM_LLM_BASE_URL", "https://opencode.ai/zen/v1")
MODEL = os.environ.get("APM_LLM_MODEL", "mimo-v2.5-free")
API_KEY = os.environ.get("APM_LLM_API_KEY", "")
CACHE_PATH = os.environ.get(
    "APM_LLM_CACHE",
    str(pathlib.Path(__file__).resolve().parents[1] / ".llm_cache.sqlite"))


def model_for(role: str | None) -> str:
    if role:
        override = os.environ.get(f"APM_LLM_MODEL_{role.upper()}")
        if override:
            return override
    return MODEL


def endpoint_for(role: str | None) -> tuple[str, str]:
    base, keys = endpoint_keys_for(role)
    return base, keys[0]


def endpoint_keys_for(role: str | None) -> tuple[str, list[str]]:
    """Per-role endpoint + KEY POOL: the key env var may hold several keys
    comma-separated; chat() rotates through them on quota errors, so daily
    limits sum across keys instead of stopping the run."""
    base, key = BASE_URL, API_KEY
    if role:
        base = os.environ.get(f"APM_LLM_BASE_URL_{role.upper()}", base)
        key = os.environ.get(f"APM_LLM_API_KEY_{role.upper()}", key)
    keys = [k.strip() for k in key.split(",") if k.strip()]
    return base, keys or [""]


# ------------------------------------------------------------------- cache --

_conn: sqlite3.Connection | None = None


def _cache() -> sqlite3.Connection | None:
    global _conn
    if not CACHE_PATH:
        return None
    if _conn is None:
        _conn = sqlite3.connect(CACHE_PATH)
        _conn.execute("CREATE TABLE IF NOT EXISTS llm_cache "
                      "(key TEXT PRIMARY KEY, response TEXT)")
        _conn.commit()
    return _conn


def _key(model, system, prompt, max_tokens, temperature,
         json_mode=False) -> str:
    # json_mode extends the key ONLY when on, so legacy cache entries for
    # plain calls stay valid
    blob = json.dumps([model, system, prompt, max_tokens, temperature]
                      + (["json"] if json_mode else []))
    return hashlib.sha256(blob.encode()).hexdigest()


# -------------------------------------------------------------------- chat --

def chat(prompt: str, system: str | None = None, max_tokens: int = 8192,
         temperature: float = 0.0, role: str | None = None,
         cacheable: bool = True, json_mode: bool = False) -> str:
    """`role` picks a per-role model; `cacheable=False` for calls whose inputs
    include evaluation labels (the judge) — hygiene, see module docstring.
    APM_LLM_MAXTOK_<ROLE> overrides max_tokens (reasoning models need headroom:
    hidden thinking + JSON output must both fit or content comes back empty).
    `json_mode` requests constrained JSON output (v1.5: whole extraction
    chunks were dying to unbalanced JSON — recall bleeding on the floor);
    endpoints that reject the param get one automatic retry without it."""
    model = model_for(role)
    if role:
        try:
            max_tokens = int(os.environ.get(f"APM_LLM_MAXTOK_{role.upper()}",
                                            max_tokens))
        except ValueError:
            pass
    if os.environ.get("APM_LLM_NO_JSON_MODE"):
        json_mode = False
    conn = _cache() if cacheable else None
    key = _key(model, system, prompt, max_tokens, temperature, json_mode)
    if conn is not None:
        row = conn.execute("SELECT response FROM llm_cache WHERE key=?",
                           (key,)).fetchone()
        if row:
            return row[0]

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    def _body(with_json: bool) -> bytes:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if with_json:
            payload["response_format"] = {"type": "json_object"}
        return json.dumps(payload).encode()
    use_json = json_mode
    base_url, key_pool = endpoint_keys_for(role)
    data = None
    n_keys = len(key_pool)
    for attempt in range(6 * n_keys):
        api_key = key_pool[attempt % n_keys]
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=_body(use_json),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}",
                     "User-Agent": "apm/0.1"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode())
            break
        except urllib.error.HTTPError as e:
            if e.code in (400, 422) and use_json:
                use_json = False        # endpoint rejects response_format
                continue
            if e.code in (403, 429, 500, 502, 503) and attempt < 6 * n_keys - 1:
                # rotate to the next key immediately; only sleep after a
                # full lap through the pool proves everyone is throttled
                if (attempt + 1) % n_keys == 0:
                    time.sleep(min(120, 8 * 2 ** (attempt // n_keys)))
                continue
            raise
    choice = data["choices"][0]
    content = choice["message"].get("content")
    if not content:
        # reasoning models can burn the whole budget thinking; surface it cleanly
        raise ValueError(f"LLM returned empty content "
                         f"(finish_reason={choice.get('finish_reason')})")
    if conn is not None:
        conn.execute("INSERT OR REPLACE INTO llm_cache VALUES (?, ?)",
                     (key, content))
        conn.commit()
    return content


if __name__ == "__main__":
    print(chat("Reply with exactly: APM transducer online", cacheable=False))
