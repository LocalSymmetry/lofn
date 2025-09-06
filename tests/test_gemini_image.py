import lofn.image_generation as ig
from lofn.ui import LofnApp
import config as root_cfg


def test_generate_image_dispatches_to_gemini(monkeypatch):
    called = {}

    def fake_generate(params, debug=False):
        called['params'] = params
        return ['img.png']

    monkeypatch.setattr(ig, 'generate_gemini_flash_image', fake_generate)
    out = ig.generate_image('Gemini 2.5 Flash Image', {'prompt': 'banana'})
    assert out == ['img.png']
    assert called['params']['prompt'] == 'banana'


def test_available_image_models_include_gemini(monkeypatch):
    monkeypatch.setattr(root_cfg.Config, 'GOOGLE_API_KEY', 'key')
    models = LofnApp.get_available_image_models(object())
    assert 'Gemini 2.5 Flash Image' in models
