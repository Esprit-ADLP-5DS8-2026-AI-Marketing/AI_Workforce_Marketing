import io
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

MAX_PIXELS = 50_000_000

ALLOWED_FORMATS = {
    "JPEG": ("image/jpeg", ".jpg"),
    "PNG": ("image/png", ".png"),
    "WEBP": ("image/webp", ".webp"),
}


class ImageValidationError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass(frozen=True)
class ValidatedImage:
    mime_type: str
    extension: str
    width: int
    height: int


def validate_image(data: bytes, max_bytes: int) -> ValidatedImage:
    """Validate by content (never by filename or client-declared type)."""
    if not data:
        raise ImageValidationError(400, "The uploaded image is empty.")
    if len(data) > max_bytes:
        raise ImageValidationError(
            413, f"The uploaded image exceeds the {max_bytes // (1024 * 1024)} MB limit."
        )
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format
            width, height = img.size
            if fmt not in ALLOWED_FORMATS:
                raise ImageValidationError(
                    415, "Unsupported image type. Allowed: JPEG, PNG, WEBP."
                )
            if width * height > MAX_PIXELS:
                raise ImageValidationError(400, "The image dimensions are too large.")
            img.load()  # detects truncated/corrupt files
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        raise ImageValidationError(400, "The uploaded file is not a valid, readable image.")

    mime_type, extension = ALLOWED_FORMATS[fmt]
    return ValidatedImage(mime_type=mime_type, extension=extension, width=width, height=height)