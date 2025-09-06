import os
import sys

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from lofn.parsing import select_best_json_candidate
from lofjson import parse_with_repairs


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


def test_multiline_string_in_json_value():
    raw = (
        "Full response from attempt 1: content='{\n"
        ' "personality_prompt": "\n'
        " # Core Strategy Framework\n"
        " ## The H.A.T.C.H. Method\n"
        " H – Hyperlocal Myth\u2011Making ...\n"
        " > \u201cYou are HB Ghost ... time.\u201d\n"
        ' "\n'
        "}' additional_kwargs={} response_metadata={}"
    )
    out = select_best_json_candidate(raw, {"personality_prompt": str})
    assert "H.A.T.C.H" in out["personality_prompt"] or "HB Ghost" in out["personality_prompt"]


def test_claude_panel_transcript():
    raw = (
        "content=[{'signature': 'abc', 'thinking': 'x'},"
        " {'text': 'json\\n{\\n  \"personality_prompt\": \"ok\"\\n}', 'type': 'text'}]"
    )
    obj, _, _ = parse_with_repairs(raw, required_keys=("personality_prompt",))
    assert obj["personality_prompt"] == "ok"
