import logging
import re
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path

from gradio_client import Client, handle_file
from PIL import Image

from .config import Settings, describe_error

log = logging.getLogger("visual_agent.flux")


class FluxError(RuntimeError):
    """FLUX generation failed."""


class FluxQuotaError(FluxError):
    """Hugging Face GPU quota exhausted; retrying will not help."""


def extract_image_path(result) -> Path:
    """Find the generated image file in the Space response (from the original script)."""
    candidates: list[str] = []

    def collect(value):
        if isinstance(value, str):
            candidates.append(value)
        elif isinstance(value, dict):
            for key in ("path", "name"):
                candidate = value.get(key)
                if isinstance(candidate, str):
                    candidates.append(candidate)
            for key in ("image", "images", "data", "result"):
                if key in value:
                    collect(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(result)

    for candidate in candidates:
        try:
            path = Path(candidate)
            if path.is_file():
                return path
        except (OSError, ValueError):
            continue

    raise FileNotFoundError(
        "Could not locate an image file in the FLUX response. "
        f"Response: {repr(result)[:500]}"
    )


class FluxService:
    def __init__(self, settings: Settings, client_factory=None):
        self._settings = settings
        self._client_factory = client_factory
        self._client = None

    def connect(self) -> None:
        s = self._settings
        factory = self._client_factory or (lambda: Client(s.flux_space, token=s.hf_token))
        try:
            self._client = factory()
        except Exception as exc:  # noqa: BLE001
            raise FluxError(
                f"Could not connect to Hugging Face Space {s.flux_space}: "
                f"{describe_error(exc, s.secret_values())}"
            ) from exc
        log.info("FLUX client initialised for %s", s.flux_space)

    def _predict_once(self, image_path: Path, prompt: str):
        s = self._settings
        # Same endpoint and arguments as the original script's predict() call,
        # submitted as a job so a timeout can be enforced.
        job = self._client.submit(
            input_image=handle_file(str(image_path.resolve())),
            prompt=prompt,
            seed=0,
            randomize_seed=True,
            guidance_scale=s.flux_guidance,
            steps=s.flux_steps,
            api_name="/infer",
        )
        try:
            return job.result(timeout=s.flux_timeout_seconds)
        except (FutureTimeoutError, TimeoutError) as exc:
            try:
                job.cancel()
            except Exception:  # noqa: BLE001
                pass
            raise FluxError(
                f"FLUX did not respond within {s.flux_timeout_seconds:.0f} seconds."
            ) from exc

    def generate(self, *, image_path: Path, prompt: str, destination: Path) -> float:
        """Generate one image, save it as PNG at `destination`, return duration."""
        if self._client is None:
            raise FluxError("FLUX client is not connected. Call connect() first.")

        s = self._settings
        secrets = s.secret_values()
        started = time.time()
        last_error: Exception | None = None

        for attempt in range(1, s.flux_max_attempts + 1):
            try:
                result = self._predict_once(image_path, prompt)
                source = extract_image_path(result)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(source) as img:
                    img.save(destination, format="PNG")
                return round(time.time() - started, 2)
            except FluxQuotaError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if re.search(r"quota|zerogpu", str(exc), re.IGNORECASE):
                    raise FluxQuotaError(
                        "Hugging Face GPU quota exhausted: " + describe_error(exc, secrets)
                    ) from exc
                log.warning(
                    "FLUX attempt %d/%d failed: %s",
                    attempt, s.flux_max_attempts, describe_error(exc, secrets),
                )
                if attempt < s.flux_max_attempts:
                    time.sleep(s.flux_retry_delay_seconds * attempt)

        if isinstance(last_error, FluxError):
            raise last_error
        raise FluxError(describe_error(last_error, secrets)) from last_error