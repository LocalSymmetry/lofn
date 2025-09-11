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

# Load only the prepare_image_messages function to avoid heavy imports
import ast
from pathlib import Path

source = Path('lofn/llm_integration.py').read_text()
module = ast.parse(source)
func_node = next(
    node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'prepare_image_messages'
)
module_ast = ast.Module(body=[func_node], type_ignores=[])
code = compile(module_ast, filename='<prepare_image_messages>', mode='exec')

class HumanMessage:
    def __init__(self, content, additional_kwargs=None):
        self.content = content
        self.additional_kwargs = additional_kwargs or {}

ns = {'HumanMessage': HumanMessage, 'List': list, 'base64': base64}
import mimetypes
ns['mimetypes'] = mimetypes
from lofn.helpers import compress_image_bytes
ns['compress_image_bytes'] = compress_image_bytes
exec(code, ns)
prepare_image_messages = ns['prepare_image_messages']


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
    part = msg.content[0]
    assert part["type"] == "image_url"
    url = part["image_url"]
    assert isinstance(url, str)
    assert url.startswith("data:image/jpeg;base64")
    assert msg.additional_kwargs == {}


def test_prepare_image_messages_handles_video():
    video_bytes = b"\x00\x00\x00\x18ftypmp42" + b"0" * 10
    b64 = base64.b64encode(video_bytes).decode()
    data_url = f"data:video/mp4;base64,{b64}"
    msgs = prepare_image_messages([data_url])
    assert len(msgs) == 1
    part = msgs[0].content[0]
    assert part["type"] == "file"
    file_data = part.get("file", {})
    assert file_data.get("mime_type") == "video/mp4"
    assert file_data.get("b64_json", "").startswith(b64[:10])
