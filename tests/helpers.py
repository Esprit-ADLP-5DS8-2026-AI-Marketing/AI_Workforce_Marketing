import io
import json
import threading
import time

from PIL import Image

from app.flux_service import FluxError
from app.schemas import CreativeConcept, ProductAnalysis

PRODUCT = {
    "product_name": "Aqua Bottle",
    "category": "Drinkware",
    "description": "A reusable steel water bottle.",
    "target_audience": "Active young adults",
    "key_benefits": ["Keeps drinks cold", "Leak-proof"],
    "visual_style": "Clean, modern",
    "advertising_platform": "Instagram",
    "format": "1:1",
    "campaign_objective": "Brand awareness",
}
HEADERS = {"X-API-Key": "test-service-key"}


def make_image_bytes(color=(10, 120, 200), fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, format=fmt)
    return buf.getvalue()


def analysis_json(n: int) -> str:
    return json.dumps({
        "visual_analysis": {"summary": "bottle"},
        "brand_analysis": {"summary": "modern"},
        "visual_constraints": ["keep label"],
        "creative_concepts": [
            {"name": f"Concept {i}", "angle": "a", "scene": "s", "composition": "c",
             "lighting": "l", "palette": ["red", "blue"], "photography": "p",
             "prompt": f"A detailed advertising photo prompt number {i}"}
            for i in range(1, n + 1)
        ],
    })


class FakeGemini:
    def __init__(self):
        self.calls = []
        self.error = None

    def analyse_product(self, *, product, image_bytes, mime_type, number_of_concepts, job_id="-"):
        self.calls.append(product.product_name)
        if self.error:
            raise self.error
        concepts = [
            CreativeConcept(
                name=f"Concept {i}", angle="a", scene="s", composition="c", lighting="l",
                palette="p", photography="ph",
                prompt=f"Ad prompt {i} for {product.product_name}",
            )
            for i in range(1, number_of_concepts + 1)
        ]
        return ProductAnalysis(
            visual_analysis={"summary": "x"}, brand_analysis={"summary": "y"},
            visual_constraints=["keep label"], creative_concepts=concepts,
            gemini_model_used="fake-model",
        )


class FakeFlux:
    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.calls: list[tuple[str, str]] = []
        self.fail_calls: set[int] = set()
        self.fail_all = False
        self._lock = threading.Lock()

    def connect(self):
        pass

    def generate(self, *, image_path, prompt, destination):
        with self._lock:
            self.calls.append((str(image_path), prompt))
            n = len(self.calls)
        if self.delay:
            time.sleep(self.delay)
        if self.fail_all or n in self.fail_calls:
            raise FluxError("simulated FLUX failure")
        Image.new("RGB", (8, 8), (200, 10, 10)).save(destination, format="PNG")
        return 0.01