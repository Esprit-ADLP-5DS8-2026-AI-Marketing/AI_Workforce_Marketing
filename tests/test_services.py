from concurrent.futures import TimeoutError as FutureTimeoutError
from types import SimpleNamespace

import pytest
from PIL import Image

from app.config import Settings
from app.flux_service import FluxError, FluxQuotaError, FluxService
from app.gemini_service import GeminiAuthError, GeminiError, GeminiService
from app.schemas import ProductInfo

from .helpers import PRODUCT, analysis_json, make_image_bytes

PRODUCT_INFO = ProductInfo(**PRODUCT)


@pytest.fixture
def product_image(tmp_path):
    path = tmp_path / "product.png"
    path.write_bytes(make_image_bytes())
    return path


# ------------------------------------------------------------------- Gemini
class APIErr(Exception):
    def __init__(self, code, msg="api error"):
        super().__init__(msg)
        self.code = code


class FakeModels:
    def __init__(self, behaviour):
        self.behaviour = behaviour  # model -> list of outcomes (consumed in order)
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append(model)
        outcome = self.behaviour[model].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)


def gemini(behaviour):
    s = Settings(gemini_api_key="k", gemini_models=("m1", "m2"), gemini_retries_per_model=2, gemini_retry_delay_seconds=0)
    models = FakeModels(behaviour)
    return GeminiService(s, client=SimpleNamespace(models=models)), models


def analyse(service, n=4):
    return service.analyse_product(product=PRODUCT_INFO, image_bytes=make_image_bytes(), mime_type="image/png", number_of_concepts=n)


def test_gemini_success():
    service, models = gemini({"m1": [analysis_json(4)]})
    result = analyse(service)
    assert result.gemini_model_used == "m1" and len(result.creative_concepts) == 4
    assert models.calls == ["m1"]


def test_gemini_404_skips_model_without_retry():
    service, models = gemini({"m1": [APIErr(404)], "m2": [analysis_json(4)]})
    assert analyse(service).gemini_model_used == "m2"
    assert models.calls == ["m1", "m2"]


def test_gemini_429_retried_then_falls_back():
    service, models = gemini({"m1": [APIErr(429), APIErr(429)], "m2": [analysis_json(4)]})
    assert analyse(service).gemini_model_used == "m2"
    assert models.calls == ["m1", "m1", "m2"]


def test_gemini_auth_error_is_fatal():
    service, models = gemini({"m1": [APIErr(403)], "m2": [analysis_json(4)]})
    with pytest.raises(GeminiAuthError):
        analyse(service)
    assert models.calls == ["m1"]


def test_gemini_bad_json_is_retried():
    service, models = gemini({"m1": ["not json", analysis_json(4)]})
    assert analyse(service).gemini_model_used == "m1"
    assert models.calls == ["m1", "m1"]


def test_gemini_too_few_concepts_then_all_fail():
    service, models = gemini({"m1": [analysis_json(2), analysis_json(2)], "m2": [analysis_json(2), analysis_json(2)]})
    with pytest.raises(GeminiError):
        analyse(service)
    assert len(models.calls) == 4  # bounded: 2 models x 2 attempts


def test_gemini_extra_concepts_truncated():
    service, _ = gemini({"m1": [analysis_json(6)]})
    assert len(analyse(service, n=4).creative_concepts) == 4


# --------------------------------------------------------------------- FLUX
class FakeJob:
    def __init__(self, outcome):
        self.outcome = outcome
        self.cancelled = False

    def result(self, timeout=None):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    def cancel(self):
        self.cancelled = True


class FakeGradioClient:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.kwargs = []
        self.jobs = []

    def submit(self, **kwargs):
        self.kwargs.append(kwargs)
        job = FakeJob(self.outcomes.pop(0))
        self.jobs.append(job)
        return job


def flux(outcomes, attempts=2):
    s = Settings(hf_token="t", flux_max_attempts=attempts, flux_retry_delay_seconds=0)
    fake = FakeGradioClient(outcomes)
    service = FluxService(s, client_factory=lambda: fake)
    service.connect()
    return service, fake


def test_flux_success_converts_to_png_and_uses_original_endpoint(tmp_path):
    source = tmp_path / "out.jpg"
    Image.new("RGB", (16, 16), (1, 2, 3)).save(source, format="JPEG")
    product = tmp_path / "product.png"
    product.write_bytes(make_image_bytes())
    service, fake = flux([[str(source), 123]])

    dest = tmp_path / "images" / "01_x_v01.png"
    service.generate(image_path=product, prompt="a prompt", destination=dest)

    with Image.open(dest) as img:
        assert img.format == "PNG"
    kw = fake.kwargs[0]
    assert kw["api_name"] == "/infer" and kw["steps"] == 28 and kw["guidance_scale"] == 2.5
    assert kw["seed"] == 0 and kw["randomize_seed"] is True and kw["prompt"] == "a prompt"


def test_flux_quota_not_retried(tmp_path, product_image):
    service, fake = flux([Exception("You have exceeded your GPU quota")])
    with pytest.raises(FluxQuotaError):
        service.generate(image_path=product_image, prompt="x", destination=tmp_path / "o.png")
    assert len(fake.kwargs) == 1


def test_flux_timeout_cancels_job(tmp_path, product_image):
    service, fake = flux([FutureTimeoutError()], attempts=1)
    with pytest.raises(FluxError):
        service.generate(image_path=product_image, prompt="x", destination=tmp_path / "o.png")
    assert fake.jobs[0].cancelled


def test_flux_retries_once_then_succeeds(tmp_path, product_image):
    source = tmp_path / "out.png"
    Image.new("RGB", (4, 4)).save(source)
    service, fake = flux([RuntimeError("temporary"), str(source)])
    service.generate(image_path=product_image, prompt="x", destination=tmp_path / "o.png")
    assert len(fake.kwargs) == 2


def test_flux_invalid_response(tmp_path, product_image):
    service, _ = flux([{"unexpected": True}], attempts=1)
    with pytest.raises(FluxError):
        service.generate(image_path=product_image, prompt="x", destination=tmp_path / "o.png")