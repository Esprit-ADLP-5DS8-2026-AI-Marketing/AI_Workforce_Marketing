# Visual Agent

A FastAPI service that turns a product photo plus product facts into advertising images.
Gemini analyses the product and writes distinct creative concepts and prompts; FLUX.1 Kontext
(Hugging Face Space, via `gradio_client`) generates the images. It is one component of a larger
marketing workforce orchestrated by n8n: n8n calls this service over HTTP. Telegram and the
upstream workflow are outside this project.

## Architecture

```
n8n ──POST /generate-creatives──▶ FastAPI (main.py) ──▶ JobManager (jobs.py, thread pool)
n8n ◀─── 202 + job_id ───────────                              │
n8n ──GET /jobs/{id} (poll)────▶                       VisualAgent (agent.py)
n8n ──GET image URLs───────────▶                        ├─ GeminiService (gemini_service.py)
                                                        └─ FluxService   (flux_service.py)
outputs/<job_id>/ ... images, creative_prompts.json, generation_report.json, job.json
```

| Module | Responsibility |
|---|---|
| `app/main.py` | HTTP layer, auth, validation, error handling, image delivery |
| `app/agent.py` | Orchestration: Gemini -> save prompts -> FLUX loop -> save report. No HTTP code. |
| `app/gemini_service.py` | Gemini client, prompt, model fallback, bounded retries, JSON validation |
| `app/flux_service.py` | Space call (`/infer`), timeout, retries, quota detection, PNG saving |
| `app/jobs.py` | Background execution and job state (memory + `job.json`) |
| `app/images.py` | Upload validation by content (type, size, pixels, corruption) |
| `app/schemas.py` | Pydantic request/response models and limits |
| `app/config.py` | `.env` loading, settings, secret redaction, logging |

**Why asynchronous?** Eight FLUX generations can take many minutes; a synchronous request would
hit n8n/proxy timeouts. The API returns immediately and n8n polls. No Redis/Celery/database is
needed: jobs run in an in-process thread pool.
Limitation: a server restart interrupts running jobs (they are reported as failed).

## Prerequisites

Windows, PowerShell, Python 3.10+, a Gemini API key, a Hugging Face token, and Docker Desktop
(or another Docker Engine with Compose).

## Setup (PowerShell, from the project root)

```powershell
.\.venv\Scripts\Activate.ps1
# If activation is blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"   # paste into SERVICE_API_KEY
notepad .env
```

Fill in `GEMINI_API_KEY`, `HF_TOKEN` and `SERVICE_API_KEY`. Never commit `.env`.

Verify which Gemini models your key can use and set `GEMINI_MODELS` accordingly:

```powershell
python -c "from dotenv import load_dotenv; load_dotenv(); import os; from google import genai; c=genai.Client(api_key=os.environ['GEMINI_API_KEY']); [print(m.name) for m in c.models.list()]"
```

## Run

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8010
```

`0.0.0.0` lets Docker containers reach the service. Allow Python through Windows Firewall
(private networks) when prompted. The API key protects the service, but do not expose port 8010
to untrusted networks.

```powershell
Invoke-RestMethod http://localhost:8010/health
```

Interactive docs: http://localhost:8010/docs

## Run with Docker Compose

Copy `.env.example` to `.env`, fill in `GEMINI_API_KEY`, `HF_TOKEN`, and a strong
`SERVICE_API_KEY`, then build and start the service:

```powershell
Copy-Item .env.example .env
docker compose up --build -d
```

The API is available at http://localhost:8010. Check the service with `docker compose ps` and
`Invoke-RestMethod http://localhost:8010/health`. Follow logs with
`docker compose logs -f visual-agent`; stop it with `docker compose down`.

Generated jobs are kept in the `visual-agent-outputs` Docker volume across container restarts.
The service uses one Uvicorn worker because job state and the work queue are in-process. To change
the host port, set `PORT` in `.env` (for example, `PORT=8011`). Do not expose the service publicly
without HTTPS and appropriate network access controls.

## Tests (no real Gemini/FLUX calls)

```powershell
python -m pytest -v
```

Manual end-to-end test (uses real Gemini and FLUX; requests only 2 images):

```powershell
$env:SERVICE_API_KEY = "<your key>"
.\scripts\manual_test.ps1 -Image products\product.jpg -ProductFile product_info.json
```

(`product_info.json` must be a flat object with the nine product fields.)

## API

All endpoints except `/health` require the header `X-API-Key: <SERVICE_API_KEY>`.

### `POST /generate-creatives`  (multipart/form-data)

| Field | Type | Description |
|---|---|---|
| `payload` | text | JSON string, see below |
| `image` | file | Product image. JPEG, PNG or WEBP, max 10 MB (`MAX_UPLOAD_MB`), validated by content |

```json
{
  "product": {
    "product_name": "Aqua Bottle",
    "category": "Drinkware",
    "description": "A reusable steel water bottle.",
    "target_audience": "Active young adults",
    "key_benefits": ["Keeps drinks cold", "Leak-proof"],
    "visual_style": "Clean, modern",
    "advertising_platform": "Instagram",
    "format": "1:1",
    "campaign_objective": "Brand awareness"
  },
  "settings": { "number_of_concepts": 4, "variations_per_concept": 2 }
}
```

`settings` is optional (defaults 4 and 2). Limits: concepts 1-6, variations 1-4, total <= 12 images.

Response `202`:
```json
{ "status": "queued", "message": "Job accepted. ...", "job_id": "3f2a...", "status_url": "http://host.docker.internal:8010/jobs/3f2a..." }
```

Errors: `401` bad key, `413` image too large, `415` unsupported image type, `400` unreadable image,
`422` invalid payload, `429` queue full, `503` server not configured. All errors look like
`{"status": "error", "message": "...", "errors": ["..."]}`.

### `GET /jobs/{job_id}`

`status` is one of `queued`, `running`, `completed`, `completed_with_errors`, `failed`.
While running, `progress` shows `completed/total`. When finished:

```json
{
  "status": "completed_with_errors",
  "message": "7/8 images generated.",
  "job_id": "3f2a...",
  "product_name": "Aqua Bottle",
  "progress": { "completed": 8, "total": 8 },
  "generated_images": [
    { "concept": "Morning Run", "concept_number": 1, "variation": 1,
      "filename": "01_morning_run_v01.png",
      "url": "http://host.docker.internal:8010/jobs/3f2a.../images/01_morning_run_v01.png",
      "duration_seconds": 41.2 }
  ],
  "creative_prompts": { "visual_analysis": {}, "brand_analysis": {}, "visual_constraints": [], "creative_concepts": [], "gemini_model_used": "..." },
  "generation_report": { "planned_images": 8, "successful_images": 7, "failed_images": 1, "failures": [], "...": "..." },
  "errors": ["Concept 3 (variation 2): FluxError: ..."]
}
```

### `GET /jobs/{job_id}/images/{filename}`

Returns the PNG. Requires the API key.

## n8n integration

### Which address should n8n use?

| n8n location | Base URL |
|---|---|
| Local n8n in Docker Desktop (Windows) | `http://host.docker.internal:8010` |
| Docker on Linux | add `--add-host=host.docker.internal:host-gateway` to the n8n container |
| n8n on the same machine without Docker | `http://localhost:8010` |
| Remote / hosted n8n | `localhost` is **not** reachable. Publish the service over HTTPS (reverse proxy or tunnel), set `PUBLIC_BASE_URL` to that URL, and keep the API key secret. |

Image URLs are built from the host of the incoming request (or `PUBLIC_BASE_URL` if set), so
URLs returned to n8n work from n8n's side.

### Create a credential

Credentials -> New -> **Header Auth**. Name: `X-API-Key`. Value: your `SERVICE_API_KEY`.

### Node 1 - HTTP Request "Submit job"

- Method: `POST`; URL: `http://host.docker.internal:8010/generate-creatives`
- Authentication: Generic Credential Type -> Header Auth -> your credential
- Send Body: on; Body Content Type: **Form-Data**
- Body parameter 1: Type **n8n Binary File**, Name `image`, Input Data Field Name = the binary property holding the Telegram photo (usually `data`)
- Body parameter 2: Type **Form Data**, Name `payload`, Value (expression):
  `{{ JSON.stringify({ product: $json.product, settings: { number_of_concepts: 4, variations_per_concept: 2 } }) }}`
  (adapt `$json.product` to where your teammate puts the nine product fields)
- Options -> Timeout: `30000` (the call returns in well under a second)

### Node 2 - Wait: 10 seconds

### Node 3 - HTTP Request "Get job"

- Method: `GET`; URL: `http://host.docker.internal:8010/jobs/{{ $('Submit job').item.json.job_id }}`
- Same Header Auth credential

### Node 4 - IF

Boolean condition: `{{ ['queued','running'].includes($json.status) }}` is true.
**true** -> back to Node 2 (Wait). **false** -> continue.
Optionally add a maximum-iterations guard (e.g. 60 loops) so a stuck job cannot loop forever.

### Node 5 - Split Out `generated_images`, then HTTP Request "Download image"

- Method: `GET`; URL: `{{ $json.url }}`; same credential
- Options -> Response -> Response Format: **File**

The binary output can be passed on to the next stage (for example the Telegram "Send Photo" node
in your teammate's workflow). Handle `completed_with_errors` (partial success) and `failed`
(`errors` array) on the n8n side.

## Image delivery and storage

Each job has an isolated directory `outputs/<job_id>/` containing `images/*.png`,
`creative_prompts.json`, `generation_report.json` and `job.json`. The uploaded input is deleted
when the job ends. Jobs older than `JOB_RETENTION_HOURS` (24 by default) are deleted on startup
and on each new submission, so n8n has that long to download the images. Filenames are generated
by the service; user-supplied filenames are never used as paths.

## Timeouts, retries and failures

- Gemini: per-request timeout `GEMINI_TIMEOUT_SECONDS`; per model `GEMINI_RETRIES_PER_MODEL`
  attempts with backoff; then the next model. 401/403 stop immediately; 400/404 skip to the next
  model; invalid JSON or the wrong concept count counts as a failed attempt.
- FLUX: each image has `FLUX_TIMEOUT_SECONDS` and `FLUX_MAX_ATTEMPTS`. A GPU-quota error is not
  retried, and the remaining images are marked as skipped.
- A failed image never discards successful ones: the job ends as `completed_with_errors`.

## Common errors

| Symptom | Likely cause / fix |
|---|---|
| `401 Invalid or missing API key` | n8n credential value differs from `SERVICE_API_KEY` |
| `503 ... missing GEMINI_API_KEY` | `.env` not filled in or server not restarted |
| Gemini 404 for every model | Model IDs not available to your key; list models (see Setup) and set `GEMINI_MODELS` |
| Gemini 429 / 503 | Quota or overload; automatic retry/fallback, otherwise wait or enable billing |
| Gemini 401/403 or "API key not valid" | Wrong key or no access |
| "GPU quota" / "ZeroGPU" | Hugging Face free quota exhausted; wait for reset, use a PRO token, or duplicate the Space |
| FLUX timeout | Space queue/cold start; raise `FLUX_TIMEOUT_SECONDS` |
| `Could not connect to Hugging Face Space` | Space sleeping/renamed or bad `HF_TOKEN`; run `inspect_flux_space.py` |
| n8n cannot connect | Wrong host (use `host.docker.internal`), service bound to 127.0.0.1, or firewall |
| Job `failed` with "interrupted by a server restart" | The server restarted during the job; resubmit |

## Security

- Never commit `.env`; it is in `.gitignore`. `.env.example` has placeholders only.
- Generation and image endpoints require `X-API-Key` (constant-time comparison; the service refuses to work if no key is configured). `/health` is open and exposes only booleans.
- Only uploaded files are accepted (no URL fetching, so no SSRF). Uploads are validated by content, size and pixel count.
- Error responses are generic or redacted; stack traces and keys never reach clients, and logs redact known secrets.
- The API key travels in a header. Use HTTPS if the service leaves your machine.
- Limitation: Starlette buffers multipart uploads before the size check runs, so put a reverse proxy
  with a body-size limit in front of any internet-facing deployment.
- Jobs are in-process; run a single uvicorn worker (the default).