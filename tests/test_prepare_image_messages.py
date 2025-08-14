from io import BytesIO
from PIL import Image
import base64
import os
from pathlib import Path

# ``llm_integration`` expects prompts under ``/lofn/prompts``. Ensure the path
# exists during testing so the module can import successfully.
prompts_src = Path(__file__).resolve().parent.parent / "lofn" / "prompts"
if not Path("/lofn/prompts").exists():
    os.makedirs("/lofn", exist_ok=True)
    os.symlink(prompts_src, "/lofn/prompts")

from lofn.llm_integration import prepare_image_messages


def make_data_url():
    img = Image.new("RGB", (100, 100), color="blue")
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def test_prepare_image_messages_inlines_jpeg():
    data_url = make_data_url()
    msgs = prepare_image_messages([data_url])
    assert len(msgs) == 1
    msg = msgs[0]
    assert isinstance(msg.content, list)
    assert msg.content[0]["type"] == "input_image"
    url = msg.content[0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64")
    assert msg.additional_kwargs == {}
