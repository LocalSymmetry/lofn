from io import BytesIO
from PIL import Image
import base64

from lofn.helpers import resize_image_to_data_url


class DummyUpload:
    def __init__(self, data: bytes, mime: str = "image/png"):
        self._data = data
        self.type = mime

    def getvalue(self) -> bytes:
        return self._data


def test_resize_image_to_data_url_limits_size_and_format():
    # Create a large red PNG image
    img = Image.new("RGB", (2000, 1500), color="red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    uploaded = DummyUpload(buf.getvalue())

    data_url = resize_image_to_data_url(uploaded)
    header, b64_data = data_url.split(",", 1)
    assert header == "data:image/jpeg;base64"

    data = base64.b64decode(b64_data)
    out_img = Image.open(BytesIO(data))
    assert max(out_img.size) == 1024
    assert out_img.format == "JPEG"
