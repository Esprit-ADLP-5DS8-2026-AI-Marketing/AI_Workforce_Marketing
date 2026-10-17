import dataclasses
import json
from urllib.parse import urlparse

from fastapi.testclient import TestClient

from app.gemini_service import GeminiError
from app.main import app, get_job_manager, get_settings

from .helpers import HEADERS, PRODUCT, make_image_bytes


def submit(client, payload=None, image_bytes=None, filename="product.png", headers=HEADERS):
    payload = {"product": PRODUCT} if payload is None else payload
    data = {"payload": payload if isinstance(payload, str) else json.dumps(payload)}
    files = {"image": (filename, make_image_bytes() if image_bytes is None else image_bytes, "image/png")}
    return client.post("/generate-creatives", data=data, files=files, headers=headers)


def run_job(client, manager, **kwargs):
    response = submit(client, **kwargs)
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    manager.wait(job_id)
    return job_id, client.get(f"/jobs/{job_id}", headers=HEADERS).json()


# ------------------------------------------------------------ basics / auth
def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["configuration"] == {"gemini_api_key": True, "hf_token": True, "service_api_key": True}
    assert "test-gemini-key" not in json.dumps(body)


def test_auth_required(client):
    assert submit(client, headers={}).status_code == 401
    assert submit(client, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/jobs/" + "0" * 32).status_code == 401


def test_unconfigured_service_key_fails_closed(client, settings):
    app.dependency_overrides[get_settings] = lambda: dataclasses.replace(settings, service_api_key="")
    assert submit(client).status_code == 503


# ----------------------------------------------------------------- validation
def test_invalid_json_payload(client):
    r = submit(client, payload="{not json")
    assert r.status_code == 422 and r.json()["status"] == "error"


def test_missing_product_field(client):
    bad = {k: v for k, v in PRODUCT.items() if k != "category"}
    r = submit(client, payload={"product": bad})
    assert r.status_code == 422
    assert any("category" in e for e in r.json()["errors"])


def test_generation_limits_enforced(client):
    r = submit(client, payload={"product": PRODUCT, "settings": {"number_of_concepts": 6, "variations_per_concept": 3}})
    assert r.status_code == 422
    r = submit(client, payload={"product": PRODUCT, "settings": {"number_of_concepts": 0}})
    assert r.status_code == 422


def test_missing_image(client):
    r = client.post("/generate-creatives", data={"payload": json.dumps({"product": PRODUCT})}, headers=HEADERS)
    assert r.status_code == 422


def test_invalid_images(client):
    assert submit(client, image_bytes=b"this is not an image").status_code == 400
    assert submit(client, image_bytes=b"").status_code == 400
    assert submit(client, image_bytes=make_image_bytes(fmt="GIF")).status_code == 415
    assert submit(client, image_bytes=make_image_bytes()[:40]).status_code == 400
    assert submit(client, image_bytes=b"0" * (1024 * 1024 + 1)).status_code == 413


# --------------------------------------------------------------------- flow
def test_success_flow(client, manager, fakes):
    job_id, body = run_job(client, manager)
    assert body["status"] == "completed"
    assert body["progress"] == {"completed": 8, "total": 8}
    assert len(body["generated_images"]) == 8
    assert len(body["creative_prompts"]["creative_concepts"]) == 4
    assert body["generation_report"]["successful_images"] == 8
    assert body["errors"] == []

    url = body["generated_images"][0]["url"]
    assert url.startswith("http://testserver/jobs/")
    image = client.get(urlparse(url).path, headers=HEADERS)
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"
    assert client.get(urlparse(url).path).status_code == 401  # images need the key too

    job_dir = manager.job_dir(job_id)
    assert (job_dir / "creative_prompts.json").is_file()
    assert (job_dir / "generation_report.json").is_file()
    assert not (job_dir / "input").exists()  # uploaded input cleaned up


def test_custom_settings(client, manager):
    _, body = run_job(client, manager, payload={"product": PRODUCT, "settings": {"number_of_concepts": 2, "variations_per_concept": 1}})
    assert len(body["generated_images"]) == 2


def test_partial_failure(client, manager, fakes):
    fakes.flux.fail_calls = {3}
    _, body = run_job(client, manager)
    assert body["status"] == "completed_with_errors"
    assert len(body["generated_images"]) == 7
    assert body["generation_report"]["failed_images"] == 1
    assert len(body["errors"]) == 1


def test_all_images_fail(client, manager, fakes):
    fakes.flux.fail_all = True
    _, body = run_job(client, manager)
    assert body["status"] == "failed"
    assert body["generated_images"] == []


def test_gemini_failure_does_not_leak_secrets(client, manager, fakes):
    fakes.gemini.error = GeminiError("boom test-gemini-key and test-hf-token")
    _, body = run_job(client, manager)
    assert body["status"] == "failed"
    assert "test-gemini-key" not in json.dumps(body)
    assert "test-hf-token" not in json.dumps(body)


def test_concurrent_jobs_are_isolated(client, manager, fakes):
    fakes.flux.delay = 0.05
    ids = {}
    for name, color in (("ProductA", (255, 0, 0)), ("ProductB", (0, 0, 255))):
        r = submit(client, payload={"product": {**PRODUCT, "product_name": name}, "settings": {"number_of_concepts": 2, "variations_per_concept": 2}}, image_bytes=make_image_bytes(color))
        assert r.status_code == 202
        ids[name] = r.json()["job_id"]
    for job_id in ids.values():
        manager.wait(job_id)

    assert ids["ProductA"] != ids["ProductB"]
    for name, job_id in ids.items():
        body = client.get(f"/jobs/{job_id}", headers=HEADERS).json()
        assert body["status"] == "completed"
        assert body["product_name"] == name
        assert all(name in c["prompt"] for c in body["creative_prompts"]["creative_concepts"])
        assert len(list((manager.job_dir(job_id) / "images").glob("*.png"))) == 4

    for image_path, prompt in fakes.flux.calls:  # each prompt used its own job's image
        owner = "ProductA" if "ProductA" in prompt else "ProductB"
        assert ids[owner] in image_path


# ------------------------------------------------------------ error handling
def test_unknown_and_malformed_job_ids(client):
    assert client.get("/jobs/" + "a" * 32, headers=HEADERS).status_code == 404
    assert client.get("/jobs/not-an-id", headers=HEADERS).status_code == 404
    assert client.get(f"/jobs/{'a' * 32}/images/x.png", headers=HEADERS).status_code == 404


def test_image_path_traversal_blocked(client, manager):
    job_id, _ = run_job(client, manager)
    for name in ("..%2Fjob.json", "..%5Cjob.json", "job.json", "01_x.png.exe"):
        assert client.get(f"/jobs/{job_id}/images/{name}", headers=HEADERS).status_code == 404


def test_unhandled_error_is_generic(client):
    class Boom:
        def submit(self, *a, **k):
            raise RuntimeError("secret internal detail")

    app.dependency_overrides[get_job_manager] = lambda: Boom()
    safe_client = TestClient(app, raise_server_exceptions=False)
    r = submit(safe_client)
    assert r.status_code == 500
    assert "secret internal detail" not in r.text
    assert "Traceback" not in r.text