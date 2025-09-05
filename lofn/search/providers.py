from typing import Dict, List, Tuple


def openai_search_answer(question: str, prefer_gpt5: bool = True) -> Tuple[str, List[Dict]]:
    """Returns (answer_text, citations[]) for OpenAI."""
    from openai import OpenAI
    client = OpenAI()

    # Try GPT-5 + Responses hosted tool first
    model = "gpt-5" if prefer_gpt5 else "gpt-4o"
    tool_type_candidates = ["web_search", "web_search_preview"]

    last_err = None
    for tool_type in tool_type_candidates:
        try:
            r = client.responses.create(
                model=model,
                input=question,
                tools=[{"type": tool_type}],
            )
            # Collect citations (Responses API returns URL annotations in output items)
            cites = []
            for item in getattr(r, "output", []) or []:
                for blk in getattr(item, "content", []) or []:
                    for ann in getattr(blk, "annotations", []) or []:
                        if getattr(ann, "type", "") == "url_citation":
                            cites.append({"title": ann.title, "url": ann.url})
            return r.output_text, cites
        except Exception as e:
            last_err = e

    # Fallback: Chat Completions with -search-preview model
    try:
        c = client.chat.completions.create(
            model="gpt-4o-search-preview",
            web_search_options={},
            messages=[{"role": "user", "content": question}],
        )
        return c.choices[0].message.content, []  # model already inlines links
    except Exception as e:
        raise RuntimeError(
            f"OpenAI search failed; last Responses error={last_err}, chat err={e}"
        )


def gemini_search_answer(question: str) -> Tuple[str, List[Dict]]:
    """Returns (answer_text, citations[]) for Gemini 2.5 Pro with Google Search grounding."""
    from google import genai
    from google.genai import types

    client = genai.Client()
    tool = types.Tool(google_search=types.GoogleSearch())
    cfg = types.GenerateContentConfig(tools=[tool])
    resp = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=question,
        config=cfg,
    )
    cites = []
    gm = getattr(resp, "grounding_metadata", None)
    if gm and getattr(gm, "grounding_chunks", None):
        for ch in gm.grounding_chunks:
            if getattr(ch, "web", None) and ch.web.uri:
                cites.append({"title": getattr(ch.web, "title", None), "url": ch.web.uri})
    return resp.text, cites
