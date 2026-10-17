import logging
import re
import secrets
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .agent import VisualAgent
from .config import Settings, setup_logging
from .images import ImageValidationError, validate_image
from .jobs import JOB_ID_RE, JobManager, JobRecord, QueueFullError
from .schemas import (
    GenerateRequest,
    JobAcceptedResponse,
    JobStatusResponse,
    Progress,
)

log = logging.getLogger("visual_agent.api")

SAFE_IMAGE_NAME = re.compile(r"^[A-Za-z0-9_\-]+\.png$")
MAX_PAYLOAD_CHARS = 100_000


# ------------------------------------------------------------ dependencies
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


@lru_cache(maxsize=1)
def get_job_manager() -> JobManager:
    settings = get_settings()
    return JobManager(settings, VisualAgent(settings))


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.service_api_key:
        raise HTTPException(503, "Server is not configured: SERVICE_API_KEY is missing.")
    if not x_api_key or not secrets.compare_digest(
        x_api_key.encode("utf-8"), settings.service_api_key.encode("utf-8")
    ):
        raise HTTPException(401, "Invalid or missing API key.", headers={"WWW-Authenticate": "ApiKey"})


# ---------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    for name, ok in (
        ("GEMINI_API_KEY", bool(settings.gemini_api_key)),
        ("HF_TOKEN", bool(settings.hf_token)),
        ("SERVICE_API_KEY", bool(settings.service_api_key)),
    ):
        if not ok:
            log.warning("%s is not set; related requests will be rejected.", name)
    manager = get_job_manager()
    manager.cleanup_expired()
    log.info("Visual Agent started. Output directory: %s", settings.output_dir)
    yield
    manager.shutdown()


app = FastAPI(title="Visual Agent", version=__version__, lifespan=lifespan)


def _error(status_code: int, message: str, errors: list[str] | None = None, headers=None):
    return JSONResponse(
        status_code=status_code,
        content={"status": "error", "message": message, "errors": errors or []},
        headers=headers,
    )


def _format_errors(errors) -> list[str]:
    out = []
    for err in errors:
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
        msg = err.get("msg", "Invalid value")
        out.append(f"{loc}: {msg}" if loc else msg)
    return out


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if isinstance(exc.detail, list):
        return _error(exc.status_code, "Request validation failed.", [str(e) for e in exc.detail], exc.headers)
    return _error(exc.status_code, str(exc.detail), headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return _error(422, "Request validation failed.", _format_errors(exc.errors()))


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return _error(500, "Internal server error.")


def _base_url(request: Request, settings: Settings) -> str:
    return (settings.public_base_url or str(request.base_url)).rstrip("/")


def _to_status_response(record: JobRecord, base_url: str) -> JobStatusResponse:
    report = record.report
    images = []
    if report:
        images = [
            img.model_copy(update={"url": f"{base_url}/jobs/{record.job_id}/images/{img.filename}"})
            for img in report.generated_images
        ]
        report = report.model_copy(update={"generated_images": images})
    return JobStatusResponse(
        status=record.status,
        message=record.message,
        job_id=record.job_id,
        product_name=record.product_name,
        progress=Progress(completed=record.progress_completed, total=record.progress_total),
        generated_images=images,
        creative_prompts=record.creative_prompts,
        generation_report=report,
        errors=record.errors,
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
    )


# ------------------------------------------------------------ endpoints
@app.get("/health")
def health(settings: Settings = Depends(get_settings)):
    return {
        "status": "ok",
        "service": "visual-agent",
        "version": __version__,
        "configuration": {  # booleans only; never values
            "gemini_api_key": bool(settings.gemini_api_key),
            "hf_token": bool(settings.hf_token),
            "service_api_key": bool(settings.service_api_key),
        },
    }


@app.post(
    "/generate-creatives",
    response_model=JobAcceptedResponse,
    status_code=202,
    dependencies=[Depends(require_api_key)],
)
async def generate_creatives(
    request: Request,
    payload: str = Form(..., description="JSON: {product: {...}, settings: {...}}"),
    image: UploadFile = File(..., description="Product reference image (JPEG/PNG/WEBP)"),
    settings: Settings = Depends(get_settings),
    manager: JobManager = Depends(get_job_manager),
):
    missing = settings.missing_for_generation()
    if missing:
        raise HTTPException(503, "Server is not configured: missing " + ", ".join(missing) + ".")

    if len(payload) > MAX_PAYLOAD_CHARS:
        raise HTTPException(413, "The payload field is too large.")
    try:
        generate_request = GenerateRequest.model_validate_json(payload)
    except ValidationError as exc:
        raise HTTPException(
            422, detail=_format_errors(exc.errors(include_url=False, include_context=False, include_input=False))
        )

    data = await image.read(settings.max_upload_bytes + 1)
    try:
        validated = await run_in_threadpool(validate_image, data, settings.max_upload_bytes)
    except ImageValidationError as exc:
        raise HTTPException(exc.status_code, exc.message)

    try:
        record = await run_in_threadpool(
            manager.submit, generate_request, data, validated.extension, validated.mime_type
        )
    except QueueFullError as exc:
        raise HTTPException(429, str(exc), headers={"Retry-After": "60"})

    return JobAcceptedResponse(
        message="Job accepted. Poll status_url until status is completed, "
        "completed_with_errors or failed.",
        job_id=record.job_id,
        status_url=f"{_base_url(request, settings)}/jobs/{record.job_id}",
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse, dependencies=[Depends(require_api_key)])
def get_job(
    job_id: str,
    request: Request,
    settings: Settings = Depends(get_settings),
    manager: JobManager = Depends(get_job_manager),
):
    record = manager.get(job_id)
    if record is None:
        raise HTTPException(404, "Job not found (it may have expired).")
    return _to_status_response(record, _base_url(request, settings))


@app.get("/jobs/{job_id}/images/{filename}", dependencies=[Depends(require_api_key)])
def get_job_image(job_id: str, filename: str, manager: JobManager = Depends(get_job_manager)):
    if not JOB_ID_RE.match(job_id) or not SAFE_IMAGE_NAME.match(filename):
        raise HTTPException(404, "Image not found.")
    path = manager.job_dir(job_id) / "images" / filename
    if not path.is_file():
        raise HTTPException(404, "Image not found.")
    return FileResponse(path, media_type="image/png")