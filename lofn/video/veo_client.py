import time
from typing import Optional
from google import genai


def veo_generate_video(
    prompt: str,
    image_bytes: Optional[bytes] = None,
    use_veo3_preview: bool = False,
):
    """Generate a video via Google's Veo models.

    Args:
        prompt: Text prompt to drive generation.
        image_bytes: Optional starting image for Veo 2 image-to-video.
        use_veo3_preview: When True and no image provided, use Veo 3 preview model.

    Returns:
        Operation response containing generated video metadata.
    """
    client = genai.Client()
    if image_bytes:
        op = client.models.generate_videos(
            model="veo-2.0-generate-001", prompt=prompt, image=image_bytes
        )
    else:
        model_name = (
            "veo-3.0-generate-preview" if use_veo3_preview else "veo-3.0-generate-001"
        )
        op = client.models.generate_videos(model=model_name, prompt=prompt)

    while not op.done:
        time.sleep(5)
        op = client.operations.get(op)
    return op.response
