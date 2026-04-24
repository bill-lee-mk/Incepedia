"""OpenRouter async batch client for Incepedia MVP generation.

Design goals:
- High concurrency via asyncio.Semaphore (benchmark-driven; not globally rate-limited)
- Automatic 429 backoff with Retry-After header honoring
- Tenacity-style exponential retry for 5xx and network errors
- Accurate per-request cost accounting from OpenRouter usage fields
- Zero global state (all config injected) — safe to use in scripts
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("incepedia.generation.client")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class GenerationResult:
    """One successful generation + metadata (schema-aligned with C11)."""

    doc_id: str
    text: str
    generator: str
    prompt_tokens: int
    completion_tokens: int
    total_cost_usd: float
    latency_s: float
    temperature: float
    attempts: int = 1
    error: str | None = None


@dataclass
class ClientStats:
    """Rolling stats, updated from successful responses."""

    n_ok: int = 0
    n_err: int = 0
    n_429: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    start_ts: float = field(default_factory=time.time)

    def snapshot(self) -> dict[str, Any]:
        dur = time.time() - self.start_ts
        return {
            "n_ok": self.n_ok,
            "n_err": self.n_err,
            "n_429": self.n_429,
            "elapsed_s": round(dur, 1),
            "rps": round(self.n_ok / max(dur, 1), 2),
            "tok_out": self.total_completion_tokens,
            "tok_per_s": round(self.total_completion_tokens / max(dur, 1), 1),
            "cost_usd": round(self.total_cost_usd, 4),
        }


class OpenRouterClient:
    """Async OpenRouter client with bounded concurrency and retry.

    Usage:
        client = OpenRouterClient(model="deepseek/deepseek-chat", concurrency=64)
        async with client:
            result = await client.generate(doc_id, prompt_messages, temperature=0.7, max_tokens=2000)
    """

    def __init__(
        self,
        model: str,
        *,
        concurrency: int = 64,
        timeout_s: float = 180.0,
        max_retries: int = 6,
        base_backoff_s: float = 2.0,
        max_backoff_s: float = 60.0,
        api_key_env: str = "OPENROUTER_API_KEY",
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.concurrency = concurrency
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.base_backoff_s = base_backoff_s
        self.max_backoff_s = max_backoff_s
        key = os.getenv(api_key_env) or os.getenv("OpenRouter_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(f"{api_key_env} not set (checked OpenRouter_API_KEY, OPENAI_API_KEY too)")
        self._api_key = key
        self._headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/bill-lee-mk/Incepedia",
            "X-Title": "Incepedia synthetic pretraining data",
            **(extra_headers or {}),
        }
        self._sem = asyncio.Semaphore(concurrency)
        self._client: httpx.AsyncClient | None = None
        self.stats = ClientStats()

    async def __aenter__(self) -> "OpenRouterClient":
        limits = httpx.Limits(
            max_connections=self.concurrency * 2,
            max_keepalive_connections=self.concurrency,
        )
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout_s, connect=15.0),
            limits=limits,
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def generate(
        self,
        doc_id: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 2500,
    ) -> GenerationResult:
        """Generate one completion; retries automatically on 429/5xx/network errors."""
        assert self._client is not None, "use `async with client:` context"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "usage": {"include": True},
        }
        t0 = time.time()
        attempt = 0
        last_err: str | None = None
        async with self._sem:
            while attempt < self.max_retries:
                attempt += 1
                try:
                    r = await self._client.post(OPENROUTER_URL, headers=self._headers, json=payload)
                    if r.status_code == 200:
                        data = r.json()
                        choice = data["choices"][0]
                        text = choice["message"]["content"] or ""
                        usage = data.get("usage", {}) or {}
                        cost = float(usage.get("cost", 0.0) or 0.0)
                        pt = int(usage.get("prompt_tokens", 0) or 0)
                        ct = int(usage.get("completion_tokens", 0) or 0)
                        self.stats.n_ok += 1
                        self.stats.total_prompt_tokens += pt
                        self.stats.total_completion_tokens += ct
                        self.stats.total_cost_usd += cost
                        return GenerationResult(
                            doc_id=doc_id,
                            text=text,
                            generator=self.model,
                            prompt_tokens=pt,
                            completion_tokens=ct,
                            total_cost_usd=cost,
                            latency_s=round(time.time() - t0, 3),
                            temperature=temperature,
                            attempts=attempt,
                        )
                    elif r.status_code == 429:
                        self.stats.n_429 += 1
                        retry_after = r.headers.get("retry-after") or r.headers.get("Retry-After")
                        wait = float(retry_after) if retry_after else min(
                            self.base_backoff_s * (2 ** (attempt - 1)), self.max_backoff_s
                        )
                        last_err = f"429 rate limit; retry_after={retry_after}"
                        log.warning("[doc=%s attempt=%d] 429 rate-limited; sleeping %.1fs", doc_id, attempt, wait)
                        await asyncio.sleep(wait)
                        continue
                    elif 500 <= r.status_code < 600:
                        wait = min(self.base_backoff_s * (2 ** (attempt - 1)), self.max_backoff_s)
                        last_err = f"{r.status_code} {r.text[:200]}"
                        log.warning("[doc=%s attempt=%d] %s; sleeping %.1fs", doc_id, attempt, last_err, wait)
                        await asyncio.sleep(wait)
                        continue
                    else:
                        # 4xx other than 429 = permanent
                        last_err = f"{r.status_code} {r.text[:200]}"
                        break
                except (httpx.TimeoutException, httpx.NetworkError) as e:
                    wait = min(self.base_backoff_s * (2 ** (attempt - 1)), self.max_backoff_s)
                    last_err = f"{type(e).__name__}: {e}"
                    log.warning("[doc=%s attempt=%d] %s; sleeping %.1fs", doc_id, attempt, last_err, wait)
                    await asyncio.sleep(wait)
                    continue
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
                    log.exception("[doc=%s attempt=%d] unexpected: %s", doc_id, attempt, last_err)
                    break
        self.stats.n_err += 1
        return GenerationResult(
            doc_id=doc_id,
            text="",
            generator=self.model,
            prompt_tokens=0,
            completion_tokens=0,
            total_cost_usd=0.0,
            latency_s=round(time.time() - t0, 3),
            temperature=temperature,
            attempts=attempt,
            error=last_err or "unknown",
        )
