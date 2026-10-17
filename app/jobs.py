import logging
import os
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .agent import AgentError, VisualAgent
from .config import Settings, utc_now_iso
from .schemas import GenerateRequest, GenerationReport, JobStatus

log = logging.getLogger("visual_agent.jobs")

JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
JOB_FILE = "job.json"


class QueueFullError(RuntimeError):
    pass


class JobRecord(BaseModel):
    job_id: str
    status: JobStatus
    message: str = ""
    product_name: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    progress_completed: int = 0
    progress_total: int = 0
    creative_prompts: dict[str, Any] | None = None
    report: GenerationReport | None = None
    errors: list[str] = []


class JobManager:
    def __init__(self, settings: Settings, agent: VisualAgent):
        self.settings = settings
        self.agent = agent
        self._records: dict[str, JobRecord] = {}
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=settings.max_concurrent_jobs, thread_name_prefix="job"
        )
        settings.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- paths
    def job_dir(self, job_id: str) -> Path:
        return self.settings.output_dir / job_id

    # ------------------------------------------------------- persistence
    def _persist(self, record: JobRecord) -> None:
        directory = self.job_dir(record.job_id)
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / (JOB_FILE + ".tmp")
        tmp.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, directory / JOB_FILE)

    def _update(self, job_id: str, **fields) -> None:
        with self._lock:
            record = self._records[job_id]
            for key, value in fields.items():
                setattr(record, key, value)
            self._persist(record)

    # --------------------------------------------------------------- API
    def submit(
        self, request: GenerateRequest, image_bytes: bytes, extension: str, mime_type: str
    ) -> JobRecord:
        self.cleanup_expired()
        with self._lock:
            pending = sum(1 for r in self._records.values() if r.status in ("queued", "running"))
            if pending >= self.settings.max_pending_jobs:
                raise QueueFullError("Too many jobs are queued. Try again later.")
            job_id = uuid.uuid4().hex
            record = JobRecord(
                job_id=job_id,
                status="queued",
                message="Job queued.",
                product_name=request.product.product_name,
                created_at=utc_now_iso(),
                progress_total=request.settings.total_images,
            )
            self._records[job_id] = record

        input_dir = self.job_dir(job_id) / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        image_path = input_dir / f"product{extension}"
        image_path.write_bytes(image_bytes)

        with self._lock:
            self._persist(record)
            self._futures[job_id] = self._executor.submit(
                self._run, job_id, request, image_path, mime_type
            )
            snapshot = record.model_copy(deep=True)
        log.info("[%s] Job queued for product %r", job_id, request.product.product_name)
        return snapshot

    def get(self, job_id: str) -> JobRecord | None:
        if not JOB_ID_RE.match(job_id):
            return None
        with self._lock:
            record = self._records.get(job_id)
            if record is not None:
                return record.model_copy(deep=True)

        path = self.job_dir(job_id) / JOB_FILE
        if not path.is_file():
            return None
        try:
            record = JobRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if record.status in ("queued", "running"):
            record.status = "failed"
            record.message = "The job was interrupted by a server restart."
            record.errors = [record.message]
        return record

    def wait(self, job_id: str, timeout: float = 30.0) -> None:
        future = self._futures.get(job_id)
        if future is not None:
            future.result(timeout=timeout)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def cleanup_expired(self) -> None:
        retention = self.settings.job_retention_hours
        if retention <= 0:
            return
        cutoff = time.time() - retention * 3600
        try:
            children = list(self.settings.output_dir.iterdir())
        except OSError:
            return
        for child in children:
            if not (child.is_dir() and JOB_ID_RE.match(child.name)):
                continue
            try:
                marker = child / JOB_FILE
                mtime = marker.stat().st_mtime if marker.exists() else child.stat().st_mtime
                with self._lock:
                    record = self._records.get(child.name)
                    if record and record.status in ("queued", "running"):
                        continue
                    if mtime < cutoff:
                        shutil.rmtree(child, ignore_errors=True)
                        self._records.pop(child.name, None)
                        self._futures.pop(child.name, None)
                        log.info("[%s] Expired job removed", child.name)
            except OSError:
                continue

    # ------------------------------------------------------------ worker
    def _run(self, job_id: str, request: GenerateRequest, image_path: Path, mime_type: str) -> None:
        self._update(
            job_id, status="running", started_at=utc_now_iso(),
            message="Analysing product and generating images.",
        )

        def on_progress(done: int, total: int) -> None:
            self._update(
                job_id, progress_completed=done, progress_total=total,
                message=f"{done}/{total} images processed.",
            )

        try:
            result = self.agent.run(
                job_id=job_id,
                product=request.product,
                options=request.settings,
                image_path=image_path,
                mime_type=mime_type,
                job_dir=self.job_dir(job_id),
                on_progress=on_progress,
            )
        except AgentError as exc:
            log.error("[%s] Job failed: %s", job_id, exc)
            self._update(
                job_id, status="failed", message=str(exc), errors=[str(exc)],
                finished_at=utc_now_iso(),
            )
        except Exception:  # noqa: BLE001
            log.exception("[%s] Unexpected job failure", job_id)
            msg = "Unexpected internal error. See server logs."
            self._update(job_id, status="failed", message=msg, errors=[msg], finished_at=utc_now_iso())
        else:
            report = result.report
            errors = [f"{f.concept} (variation {f.variation}): {f.error}" for f in report.failures]
            if report.successful_images == 0:
                status, message = "failed", "No images were generated successfully."
            elif report.failed_images:
                status = "completed_with_errors"
                message = f"{report.successful_images}/{report.planned_images} images generated."
            else:
                status = "completed"
                message = f"All {report.planned_images} images generated."
            self._update(
                job_id, status=status, message=message, errors=errors,
                report=report, creative_prompts=result.analysis.model_dump(mode="json"),
                progress_completed=report.planned_images, finished_at=utc_now_iso(),
            )
        finally:
            shutil.rmtree(image_path.parent, ignore_errors=True)  # delete the uploaded input