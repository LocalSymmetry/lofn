import base64
import io
from PIL import Image, ImageOps

SUPPORTED_MIME = {"image/png": "png", "image/jpeg": "jpeg", "image/webp": "webp"}

def normalize_image_bytes(raw_bytes: bytes, max_side: int = 2048) -> tuple[bytes, str]:
    """Normalize orientation and size then encode as PNG bytes.

    Args:
        raw_bytes: Original image bytes.
        max_side: Maximum width or height after resizing.

    Returns:
        Tuple of (png_bytes, mime_type)
    """
    img = Image.open(io.BytesIO(raw_bytes))
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue(), "image/png"

def to_data_url(image_bytes: bytes, mime: str) -> str:
    """Return a data URL for the given image bytes."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"
