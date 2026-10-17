import json
import logging
import re
import time

from google import genai
from google.genai import types

from .config import Settings, describe_error
from .schemas import ProductAnalysis, ProductInfo

log = logging.getLogger("visual_agent.gemini")

AUTH_CODES = (401, 403)
# Retrying the same model will not help; move to the next one.
SKIP_MODEL_CODES = (400, 404)

# Prompt preserved from the original visual_agent.py.
#  INSTRUCTIONS_TEMPLATE = """
# You are a professional advertising creative director.

# Analyse the attached product image and the supplied product facts.

# PRODUCT INFORMATION:
# {product_json}

# Return one valid JSON object with exactly these top-level keys:

# 1. visual_analysis
#    Describe observable appearance, shape, packaging, colors,
#    materials, and other visible characteristics.
#    Do not claim to see details that are not visible.

# 2. brand_analysis
#    Explain the advertising direction using the product facts
#    and requested visual style.

# 3. visual_constraints
#    Return a list of important details to preserve, including
#    product shape, packaging, colors, and visible branding.

# 4. creative_concepts
#    Return exactly {number_of_concepts} distinct concepts.

# Every concept must contain:
# - name
# - angle
# - scene
# - composition
# - lighting
# - palette
# - photography
# - prompt

# CREATIVE REQUIREMENTS:
# - Make the concepts meaningfully different.
# - Respect the target audience and campaign objective.
# - Follow the provided brand style and advertising platform.
# - Follow the requested image format.
# - Make the result look like professional commercial photography.
# - Keep the product as the main visual focus.
# - Do not invent benefits, certifications, or unsupported claims.
# - Do not invent brand names, logos, labels, or packaging text.
# - Avoid unnecessary text inside generated images.
# - Preserve product identity as much as the image model allows.
# - Provide detailed prompts suitable for image generation.
# - Return JSON only, with no Markdown fences.
# """

INSTRUCTIONS_TEMPLATE = """
You are a senior art director who shoots real skincare campaigns for Instagram.
Your images must look like genuine photographs taken by a professional with a real
camera, never like AI art, 3D renders or stock templates.

Analyse the attached product photo and the supplied product facts.

PRODUCT INFORMATION:
{product_json}

Return one valid JSON object with exactly these top-level keys:

1. visual_analysis
   Describe only what is visible: container type (bottle, jar, tube, dropper,
   pump), shape, proportions, color, material, cap, and label layout.
   Do not claim to see details that are not visible.

2. brand_analysis
   Explain the advertising direction from the product facts and requested style.

3. visual_constraints
   Return a list of details that must never change: shape, proportions, color,
   finish, cap or pump, label position, and any visible text or logo.

4. creative_concepts
   Return exactly {number_of_concepts} concepts, in this exact order. If fewer than
   4 are requested, use the first ones from this list. Use these exact names:

   Concept 1, name "Luxury Hero":
   The product alone on one premium surface (honed marble, travertine, dark slate
   or a stone plinth). One soft directional light from the side, a gentle natural
   shadow, a faint soft reflection at most. Calm, rich, restrained palette.
   Camera slightly below eye level, 85mm lens, shallow depth of field.
   No people, no clutter.

   Concept 2, name "Lifestyle":
   A real morning skincare moment. The product stands on a vanity or bathroom
   shelf in the sharp foreground. Behind it, softly out of focus, a woman is
   touching her cheek or neck with her fingertips as if applying skincare.
   Crop the frame so her eyes are NOT visible: show only the lower half of the
   face, jaw, neck and shoulder, or just a hand and cheek. Warm natural window
   light, linen or towel texture, one small plant at most. Authentic and calm.
   The person must never hold the product. Natural skin with visible pores and
   small imperfections, relaxed natural hands with five fingers, no plastic or
   airbrushed skin.

   Concept 3, name "Minimalist Studio":
   The product on a seamless matte backdrop in one soft color that complements the
   packaging. At most one simple geometric prop (sphere, arch or block), clearly
   smaller than the product. Soft studio light from one side, a crisp but gentle
   shadow, generous empty space. No people.

   Concept 4, name "Botanical Editorial":
   The product on a natural surface (stone, wood or wet slate) with a few fresh
   botanicals that match the product facts, such as leaves, citrus slices, aloe or
   herbs. Props are arranged naturally and sit lower or beside the product, never
   covering it. A few small water droplets at most. Soft natural daylight, earthy
   green palette, layered depth. No people.

Every concept must contain: name, angle, scene, composition, lighting, palette,
photography, prompt.

RULES FOR THE prompt FIELD (it is sent to an image-editing model):
- Write 4 to 6 short, concrete, literal sentences. No poetry, no story.
- Sentence 1, always: Keep the product exactly as it appears in the photo, with
  the same shape, size, color, material, label and text, unchanged and fully
  visible.
- Then describe the new setting, surface and props in plain words.
- Then describe the light: its direction and softness, and that shadows and
  reflections match it.
- Then describe the camera: lens in millimeters, height and angle, shallow depth
  of field, a slight natural film grain.
- Compose for a vertical 4:5 Instagram feed image: product near the center, with
  clear margins on all sides.
- Make it physically plausible: objects rest on surfaces, nothing floats, props
  are realistic in size, the product keeps its real-world scale.
- Last sentence, always: No added text, no extra logos, no watermarks, nothing
  covering the product label.
- Do NOT use hype words such as stunning, ultra-realistic, hyper-detailed, 8k,
  masterpiece, cinematic, perfect, flawless or dramatic.
- Do not stack many props. Fewer, well-chosen elements look more real.
- Never invent benefits, certifications, claims or ingredients that are not in
  the product facts.
- Keep each palette cohesive and make the product the clear focus.

Return JSON only, with no Markdown fences.
"""
class GeminiError(RuntimeError):
    """Gemini could not produce a usable analysis."""


class GeminiAuthError(GeminiError):
    """Authentication/permission problem; retrying cannot help."""


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    return re.sub(r"\s*```$", "", text).strip()


class GeminiService:
    def __init__(self, settings: Settings, client=None):
        self._settings = settings
        self._client = client

    def _get_client(self):
        if self._client is None:
            if not self._settings.gemini_api_key:
                raise GeminiAuthError("GEMINI_API_KEY is not configured.")
            self._client = genai.Client(
                api_key=self._settings.gemini_api_key,
                http_options=types.HttpOptions(
                    timeout=int(self._settings.gemini_timeout_seconds * 1000)
                ),
            )
        return self._client

    @staticmethod
    def _parse(text: str | None, number_of_concepts: int) -> ProductAnalysis:
        if not text:
            raise ValueError("Gemini returned an empty response.")
        try:
            data = json.loads(_strip_fences(text))
        except json.JSONDecodeError as exc:
            raise ValueError("Gemini response is not valid JSON.") from exc
        analysis = ProductAnalysis.model_validate(data)
        if len(analysis.creative_concepts) < number_of_concepts:
            raise ValueError(
                f"Expected {number_of_concepts} concepts, "
                f"received {len(analysis.creative_concepts)}."
            )
        analysis.creative_concepts = analysis.creative_concepts[:number_of_concepts]
        return analysis

    def analyse_product(
        self,
        *,
        product: ProductInfo,
        image_bytes: bytes,
        mime_type: str,
        number_of_concepts: int,
        job_id: str = "-",
    ) -> ProductAnalysis:
        s = self._settings
        secrets = s.secret_values()
        client = self._get_client()

        instructions = INSTRUCTIONS_TEMPLATE.format(
            product_json=json.dumps(product.model_dump(), ensure_ascii=False, indent=2),
            number_of_concepts=number_of_concepts,
        )
        contents = [instructions, types.Part.from_bytes(data=image_bytes, mime_type=mime_type)]
        config = types.GenerateContentConfig(response_mime_type="application/json")

        errors: list[str] = []
        total_models = len(s.gemini_models)

        for model_index, model in enumerate(s.gemini_models, start=1):
            for attempt in range(1, s.gemini_retries_per_model + 1):
                log.info(
                    "[%s] Gemini model %d/%d %s, attempt %d/%d",
                    job_id, model_index, total_models, model, attempt, s.gemini_retries_per_model,
                )
                try:
                    response = client.models.generate_content(
                        model=model, contents=contents, config=config
                    )
                    analysis = self._parse(response.text, number_of_concepts)
                    analysis.gemini_model_used = model
                    log.info("[%s] Gemini model succeeded: %s", job_id, model)
                    return analysis
                except Exception as exc:  # noqa: BLE001 - classified below
                    code = getattr(exc, "code", None)
                    summary = describe_error(exc, secrets)
                    errors.append(f"{model} (attempt {attempt}): {summary}")
                    log.warning("[%s] Gemini failure on %s: %s", job_id, model, summary)

                    if code in AUTH_CODES or "api key not valid" in str(exc).lower():
                        raise GeminiAuthError(
                            "Gemini authentication or permission error. "
                            "Check GEMINI_API_KEY and account access."
                        ) from exc
                    if code in SKIP_MODEL_CODES:
                        break  # model unavailable / bad request: next model
                    if attempt < s.gemini_retries_per_model:
                        delay = s.gemini_retry_delay_seconds * attempt
                        log.info("[%s] Retrying %s in %.1fs", job_id, model, delay)
                        time.sleep(delay)

            if model_index < total_models:
                log.warning("[%s] Falling back to %s", job_id, s.gemini_models[model_index])

        raise GeminiError(
            "All configured Gemini models failed. Check API access, quota, billing "
            "and model availability. Last errors: " + " | ".join(errors[-3:])
        )