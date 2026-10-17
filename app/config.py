import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Taken from the original visual_agent.py. Override with GEMINI_MODELS.
DEFAULT_GEMINI_MODELS = (
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
)

_TOKEN_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{10,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
)


class ConfigError(RuntimeError):
    """Raised when configuration is invalid."""


def _env_number(name: str, default, cast):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return cast(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} must be a valid {cast.__name__}.") from exc


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = ""
    hf_token: str = ""
    service_api_key: str = ""

    gemini_models: tuple[str, ...] = DEFAULT_GEMINI_MODELS
    gemini_retries_per_model: int = 2
    gemini_retry_delay_seconds: float = 3.0
    gemini_timeout_seconds: float = 60.0

    flux_space: str = "black-forest-labs/FLUX.1-Kontext-Dev"
    flux_steps: int = 28
    flux_guidance: float = 2.5
    flux_timeout_seconds: float = 300.0
    flux_max_attempts: int = 2
    flux_retry_delay_seconds: float = 5.0

    output_dir: Path = BASE_DIR / "outputs"
    public_base_url: str = ""
    max_upload_mb: int = 10
    max_concurrent_jobs: int = 1
    max_pending_jobs: int = 10
    job_retention_hours: float = 24.0
    log_level: str = "INFO"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def secret_values(self) -> list[str]:
        return [
            v for v in (self.gemini_api_key, self.hf_token, self.service_api_key) if v
        ]

    def missing_for_generation(self) -> list[str]:
        missing = []
        if not self.gemini_api_key:
            missing.append("GEMINI_API_KEY")
        if not self.hf_token:
            missing.append("HF_TOKEN")
        return missing

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(BASE_DIR / ".env")

        models_raw = os.getenv("GEMINI_MODELS", "")
        models = tuple(m.strip() for m in models_raw.split(",") if m.strip())

        output_dir = Path(os.getenv("OUTPUT_DIR") or "outputs")
        if not output_dir.is_absolute():
            output_dir = BASE_DIR / output_dir

        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            hf_token=os.getenv("HF_TOKEN", "").strip(),
            service_api_key=os.getenv("SERVICE_API_KEY", "").strip(),
            gemini_models=models or DEFAULT_GEMINI_MODELS,
            gemini_retries_per_model=_env_number("GEMINI_RETRIES_PER_MODEL", 2, int),
            gemini_retry_delay_seconds=_env_number("GEMINI_RETRY_DELAY_SECONDS", 3.0, float),
            gemini_timeout_seconds=_env_number("GEMINI_TIMEOUT_SECONDS", 60.0, float),
            flux_space=os.getenv("FLUX_SPACE", "black-forest-labs/FLUX.1-Kontext-Dev").strip(),
            flux_steps=_env_number("FLUX_STEPS", 28, int),
            flux_guidance=_env_number("FLUX_GUIDANCE", 2.5, float),
            flux_timeout_seconds=_env_number("FLUX_TIMEOUT_SECONDS", 300.0, float),
            flux_max_attempts=_env_number("FLUX_MAX_ATTEMPTS", 2, int),
            flux_retry_delay_seconds=_env_number("FLUX_RETRY_DELAY_SECONDS", 5.0, float),
            output_dir=output_dir,
            public_base_url=os.getenv("PUBLIC_BASE_URL", "").strip(),
            max_upload_mb=_env_number("MAX_UPLOAD_MB", 10, int),
            max_concurrent_jobs=max(1, _env_number("MAX_CONCURRENT_JOBS", 1, int)),
            max_pending_jobs=max(1, _env_number("MAX_PENDING_JOBS", 10, int)),
            job_retention_hours=_env_number("JOB_RETENTION_HOURS", 24.0, float),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        )


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """Remove known secrets and token-shaped strings from text."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    for pattern in _TOKEN_PATTERNS:
        text = pattern.sub("***", text)
    return text


def describe_error(exc: BaseException, secrets: Iterable[str] = (), limit: int = 400) -> str:
    """One-line, redacted, length-bounded description of an exception."""
    text = redact(f"{type(exc).__name__}: {exc}", secrets)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )