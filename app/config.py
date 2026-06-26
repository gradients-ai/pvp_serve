"""Runtime settings, read once from the environment."""

import os
from dataclasses import dataclass, field


def _split_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    sglang_base_url: str = field(default_factory=lambda: os.environ.get("PVP_SGLANG_BASE_URL", "http://localhost:30000/v1"))
    inference_model: str = field(default_factory=lambda: os.environ.get("PVP_INFERENCE_MODEL", "champion"))
    temperature: float = field(default_factory=lambda: float(os.environ.get("PVP_TEMPERATURE", "0.0")))
    seed: int | None = field(default_factory=lambda: _opt_int(os.environ.get("PVP_SEED")))
    agent_kind: str = field(default_factory=lambda: os.environ.get("PVP_AGENT_KIND", "llm").lower())
    host: str = field(default_factory=lambda: os.environ.get("PVP_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.environ.get("PVP_PORT", "8000")))
    cors_origins: list[str] = field(
        default_factory=lambda: _split_csv(os.environ.get("PVP_CORS_ORIGINS", "http://localhost:5173"))
    )
    # How long a session may sit idle before it is evicted.
    session_ttl_seconds: int = field(default_factory=lambda: int(os.environ.get("PVP_SESSION_TTL_SECONDS", "3600")))
    # --- general chat with the champion ---
    chat_temperature: float = field(default_factory=lambda: float(os.environ.get("PVP_CHAT_TEMPERATURE", "0.7")))
    chat_max_tokens: int = field(default_factory=lambda: int(os.environ.get("PVP_CHAT_MAX_TOKENS", "1024")))
    chat_system_prompt: str = field(
        default_factory=lambda: os.environ.get(
            "PVP_CHAT_SYSTEM_PROMPT",
            "You are the Gradients champion — the model that won the latest Bittensor subnet 56 "
            "environment tournament. Be helpful, direct, and concise.",
        )
    )
    # Max messages accepted in a single chat request (guards context + abuse).
    chat_max_messages: int = field(default_factory=lambda: int(os.environ.get("PVP_CHAT_MAX_MESSAGES", "40")))

    @property
    def is_llm(self) -> bool:
        return self.agent_kind == "llm"


def _opt_int(value: str | None) -> int | None:
    return int(value) if value not in (None, "") else None


settings = Settings()
