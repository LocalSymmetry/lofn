import os
import sys

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from lofn.parsing import select_best_json_candidate


def test_personality_from_logged_content():
    raw = (
        "Full response from attempt 1: content='{\n "
        "\"personality_prompt\": \"Hello\\nWorld\"\n}' additional_kwargs={}"
    )
    out = select_best_json_candidate(raw, {"personality_prompt": str})
    assert out["personality_prompt"].startswith("Hello")


def test_facets_from_numbered_dict():
    raw = "content='{\"facets\": {\"0\": \"a\", \"1\": \"b\", \"2\": \"c\"}}'"
    out = select_best_json_candidate(raw, {"facets": "list[str]"})
    assert out["facets"] == ["a", "b", "c"]


def test_string_wrapped_json():
    raw = '"{\\"meta_prompt\\": \\"ok\\"}"'
    out = select_best_json_candidate(raw, {"meta_prompt": str})
    assert out["meta_prompt"] == "ok"


def test_code_fenced_json():
    raw = "```json\n{\n  \"meta_prompt\": \"ok\"\n}\n```"
    out = select_best_json_candidate(raw, {"meta_prompt": str})
    assert out["meta_prompt"] == "ok"
