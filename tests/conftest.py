from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.agent import VisualAgent
from app.config import Settings
from app.jobs import JobManager
from app.main import app, get_job_manager, get_settings

from .helpers import FakeFlux, FakeGemini


@pytest.fixture
def settings(tmp_path):
    return Settings(
        gemini_api_key="test-gemini-key",
        hf_token="test-hf-token",
        service_api_key="test-service-key",
        output_dir=tmp_path / "outputs",
        max_concurrent_jobs=2,
        max_upload_mb=1,
        gemini_retry_delay_seconds=0,
        flux_retry_delay_seconds=0,
    )


@pytest.fixture
def fakes():
    return SimpleNamespace(gemini=FakeGemini(), flux=FakeFlux())


@pytest.fixture
def manager(settings, fakes):
    agent = VisualAgent(settings, gemini=fakes.gemini, flux_factory=lambda: fakes.flux)
    m = JobManager(settings, agent)
    yield m
    m.shutdown()


@pytest.fixture
def client(settings, manager):
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_job_manager] = lambda: manager
    yield TestClient(app)  # no `with`: the lifespan (real settings) is not started
    app.dependency_overrides.clear()