"""General chat with the champion model.

A thin async proxy in front of the SGLang OpenAI-compatible endpoint so the
inference server itself stays off the public internet (CORS, a default system
persona, and basic input bounds live here). Unrelated to the game harness — no
SIGALRM, no pyspiel — so it runs straight on the event loop and does NOT take
the game turn lock.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.config import settings
from app.contract import ChatRequest

_client = AsyncOpenAI(base_url=settings.sglang_base_url.rstrip("/"), api_key="dummy", max_retries=1)


def _build_messages(req: ChatRequest) -> list[dict[str, str]]:
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    # Ensure a single leading system prompt (the server's persona) unless the
    # caller already supplied one.
    if not msgs or msgs[0]["role"] != "system":
        msgs.insert(0, {"role": "system", "content": settings.chat_system_prompt})
    return msgs


async def stream_chat(req: ChatRequest) -> AsyncIterator[str]:
    """Yield Server-Sent Events: `data: {"delta": "..."}` then `data: [DONE]`.

    Errors are surfaced as a final `data: {"error": "..."}` event so the UI can
    show them instead of the stream dying silently.
    """
    messages = _build_messages(req)
    try:
        stream = await _client.chat.completions.create(
            model=settings.inference_model,
            messages=messages,
            temperature=req.temperature if req.temperature is not None else settings.chat_temperature,
            max_tokens=req.maxTokens or settings.chat_max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield f"data: {json.dumps({'delta': delta})}\n\n"
    except Exception as exc:  # noqa: BLE001 — surface any upstream failure to the client
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"
    finally:
        yield "data: [DONE]\n\n"
