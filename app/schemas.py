from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

MAX_CONCEPTS = 6
MAX_VARIATIONS = 4
MAX_TOTAL_IMAGES = 12

JobStatus = Literal["queued", "running", "completed", "completed_with_errors", "failed"]

Benefit = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


# ---------------------------------------------------------------- requests
class ProductInfo(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    product_name: str = Field(min_length=1, max_length=200)
    category: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=3000)
    target_audience: str = Field(min_length=1, max_length=1000)
    key_benefits: list[Benefit] = Field(min_length=1, max_length=20)
    visual_style: str = Field(min_length=1, max_length=1000)
    advertising_platform: str = Field(min_length=1, max_length=200)
    format: str = Field(min_length=1, max_length=100)
    campaign_objective: str = Field(min_length=1, max_length=1000)

    @field_validator("key_benefits", mode="before")
    @classmethod
    def _benefits_as_list(cls, value):
        return [value] if isinstance(value, str) else value


class GenerationSettings(BaseModel):
    number_of_concepts: int = Field(default=4, ge=1, le=MAX_CONCEPTS)
    variations_per_concept: int = Field(default=2, ge=1, le=MAX_VARIATIONS)

    @model_validator(mode="after")
    def _limit_total(self):
        total = self.number_of_concepts * self.variations_per_concept
        if total > MAX_TOTAL_IMAGES:
            raise ValueError(
                f"number_of_concepts x variations_per_concept must be <= {MAX_TOTAL_IMAGES} (got {total})"
            )
        return self

    @property
    def total_images(self) -> int:
        return self.number_of_concepts * self.variations_per_concept


class GenerateRequest(BaseModel):
    """The JSON placed in the multipart `payload` field."""

    product: ProductInfo
    settings: GenerationSettings = Field(default_factory=GenerationSettings)


# ------------------------------------------------------- Gemini output
class CreativeConcept(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1)
    prompt: str = Field(min_length=10)
    # Gemini sometimes returns lists/objects here, so the type is not enforced.
    angle: Any = ""
    scene: Any = ""
    composition: Any = ""
    lighting: Any = ""
    palette: Any = ""
    photography: Any = ""


class ProductAnalysis(BaseModel):
    model_config = ConfigDict(extra="allow")

    visual_analysis: Any = Field(...)
    brand_analysis: Any = Field(...)
    visual_constraints: Any = Field(...)
    creative_concepts: list[CreativeConcept] = Field(min_length=1)
    gemini_model_used: str | None = None


# ------------------------------------------------------------- results
class GeneratedImage(BaseModel):
    concept: str
    concept_number: int
    variation: int
    filename: str
    url: str | None = None
    duration_seconds: float


class FailedImage(BaseModel):
    concept: str
    concept_number: int
    variation: int
    error_type: str
    error: str


class GenerationReport(BaseModel):
    job_id: str
    product_name: str
    started_at: str
    finished_at: str
    total_duration_seconds: float
    gemini_model_used: str | None
    planned_images: int
    successful_images: int
    failed_images: int
    generated_images: list[GeneratedImage]
    failures: list[FailedImage]
    creative_prompts_file: str = "creative_prompts.json"


# ------------------------------------------------------------ responses
class Progress(BaseModel):
    completed: int
    total: int


class JobAcceptedResponse(BaseModel):
    status: Literal["queued"] = "queued"
    message: str
    job_id: str
    status_url: str


class JobStatusResponse(BaseModel):
    status: JobStatus
    message: str
    job_id: str
    product_name: str
    progress: Progress
    generated_images: list[GeneratedImage] = []
    creative_prompts: dict[str, Any] | None = None
    generation_report: GenerationReport | None = None
    errors: list[str] = []
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None