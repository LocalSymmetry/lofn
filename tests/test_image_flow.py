import base64
import lofn.llm_integration as li
from lofn.llm_integration import call_openai_gpt5_multimodal
from lofn.utils.image_io import normalize_image_bytes


def test_openai_accepts_image(monkeypatch):
    tiny_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAAC0lEQVR42mP8/x8AAwMBgB6+VsQAAAAASUVORK5CYII="
    )
    img, mime = normalize_image_bytes(tiny_png, max_side=1)

    class FakeResponses:
        @staticmethod
        def create(**kwargs):
            assert kwargs["model"].startswith("gpt-5")
            inp = kwargs["input"][-1]["content"]
            assert any(x.get("type") == "input_image" for x in inp)
            return type("R", (), {"output_text": "ok"})

    class FakeClient:
        responses = FakeResponses()

    monkeypatch.setattr(li, "OpenAI", lambda: FakeClient())
    out = call_openai_gpt5_multimodal("gpt-5", "describe", [img], "sys")
    assert out == "ok"
