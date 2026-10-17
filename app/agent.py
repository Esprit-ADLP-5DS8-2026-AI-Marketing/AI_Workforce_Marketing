import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import Settings, describe_error, redact, utc_now_iso
from .flux_service import FluxError, FluxQuotaError, FluxService
from .gemini_service import GeminiError, GeminiService
from .schemas import (
    FailedImage,
    GeneratedImage,
    GenerationReport,
    GenerationSettings,
    ProductAnalysis,
    ProductInfo,
)

log = logging.getLogger("visual_agent.agent")

PROMPTS_FILENAME = "creative_prompts.json"
REPORT_FILENAME = "generation_report.json"


class AgentError(RuntimeError):
    """The job could not run (message is safe to show to API clients)."""


@dataclass
class AgentResult:
    analysis: ProductAnalysis
    report: GenerationReport


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name)
    return cleaned.strip("_").lower()[:60] or "concept"


def _write_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


class VisualAgent:
    def __init__(self, settings: Settings, gemini=None, flux_factory: Callable | None = None):
        self._settings = settings
        self._gemini = gemini or GeminiService(settings)
        self._flux_factory = flux_factory or (lambda: FluxService(settings))

    def run(
        self,
        *,
        job_id: str,
        product: ProductInfo,
        options: GenerationSettings,
        image_path: Path,
        mime_type: str,
        job_dir: Path,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> AgentResult:
        secrets = self._settings.secret_values()
        started = time.time()
        started_at = utc_now_iso()
        n_concepts = options.number_of_concepts
        n_variations = options.variations_per_concept
        total = options.total_images

        images_dir = job_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "[%s] Start: product=%r concepts=%d variations=%d total=%d",
            job_id, product.product_name, n_concepts, n_variations, total,
        )
        if on_progress:
            on_progress(0, total)

        # --- Gemini analysis -------------------------------------------------
        try:
            analysis = self._gemini.analyse_product(
                product=product,
                image_bytes=image_path.read_bytes(),
                mime_type=mime_type,
                number_of_concepts=n_concepts,
                job_id=job_id,
            )
        except GeminiError as exc:
            raise AgentError(redact(f"Gemini analysis failed: {exc}", secrets)) from exc

        log.info("[%s] Selected Gemini model: %s", job_id, analysis.gemini_model_used)
        _write_json(job_dir / PROMPTS_FILENAME, analysis.model_dump(mode="json"))

        # --- FLUX connection -------------------------------------------------
        flux = self._flux_factory()
        try:
            flux.connect()
        except FluxError as exc:
            raise AgentError(redact(str(exc), secrets)) from exc

        # --- Image generation ------------------------------------------------
        successes: list[GeneratedImage] = []
        failures: list[FailedImage] = []
        quota_exhausted = False
        attempted = 0

        for concept_number, concept in enumerate(analysis.creative_concepts, start=1):
            for variation in range(1, n_variations + 1):
                attempted += 1
                filename = (
                    f"{concept_number:02d}_{safe_filename(concept.name)}_v{variation:02d}.png"
                )
                log.info(
                    "[%s] Image %d/%d (concept %d '%s', variation %d)",
                    job_id, attempted, total, concept_number, concept.name, variation,
                )

                if quota_exhausted:
                    failures.append(FailedImage(
                        concept=concept.name, concept_number=concept_number,
                        variation=variation, error_type="Skipped",
                        error="Skipped: Hugging Face GPU quota was exhausted earlier in this job.",
                    ))
                else:
                    try:
                        duration = flux.generate(
                            image_path=image_path,
                            prompt=concept.prompt,
                            destination=images_dir / filename,
                        )
                        successes.append(GeneratedImage(
                            concept=concept.name, concept_number=concept_number,
                            variation=variation, filename=filename,
                            duration_seconds=duration,
                        ))
                        log.info("[%s] Saved %s in %.1fs", job_id, filename, duration)
                    except Exception as exc:  # noqa: BLE001 - partial success is a feature
                        if isinstance(exc, FluxQuotaError):
                            quota_exhausted = True
                        summary = describe_error(exc, secrets)
                        log.error("[%s] Image failed: %s", job_id, summary)
                        failures.append(FailedImage(
                            concept=concept.name, concept_number=concept_number,
                            variation=variation, error_type=type(exc).__name__,
                            error=summary,
                        ))

                if on_progress:
                    on_progress(attempted, total)

        # --- Report ------------------------------------------------------------
        report = GenerationReport(
            job_id=job_id,
            product_name=product.product_name,
            started_at=started_at,
            finished_at=utc_now_iso(),
            total_duration_seconds=round(time.time() - started, 2),
            gemini_model_used=analysis.gemini_model_used,
            planned_images=total,
            successful_images=len(successes),
            failed_images=len(failures),
            generated_images=successes,
            failures=failures,
            creative_prompts_file=PROMPTS_FILENAME,
        )
        _write_json(job_dir / REPORT_FILENAME, report.model_dump(mode="json"))

        log.info(
            "[%s] Finished: %d/%d succeeded, %d failed, %.1fs",
            job_id, len(successes), total, len(failures), report.total_duration_seconds,
        )
        return AgentResult(analysis=analysis, report=report)