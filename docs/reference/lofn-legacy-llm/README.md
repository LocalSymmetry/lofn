# Legacy LLM integration — reference for the Lofn Creative Studio build

Vendored 2026-07-03 from the operator's streamlit-era checkout (`E:\Lofn_test\lofn`),
the fossil ancestor of the Studio execution engine. **Reference only — do not import
or wire into production.** It shows *how each provider was actually integrated* so the
fleet ports proven patterns instead of guessing. Keys in `config.yaml` are empty placeholders.

| File | What it teaches | Informs |
|---|---|---|
| `llm_integration.py` | **The engine ancestor.** Per-provider adapters: `OpenRouterLLM` (raw `requests` → `https://openrouter.ai/api/v1/chat/completions`, Bearer auth; `fetch_openrouter_models()` for context-length discovery; `OR-` prefix), `PoeLLM` (**`fastapi_poe`** — `fp.get_bot_response(bot_name=<model minus "Poe-">, api_key=…)`, collect `PartialResponse.text`, **no stop sequences** — Poe is NOT OpenAI-compatible), `GeminiLLM` (native `google.genai` `generate_content` + 2.5 thinking budget + optional GoogleSearch tool), OpenAI (`ChatOpenAIWebSearch` via Responses API, `O1ChatOpenAI` for reasoning models), `LOCAL-` (ChatOpenAI w/ custom base_url). `get_llm()` = the provider router by model-name prefix. `run_chain_with_retries()` = the **retry-with-correction loop** (parse → on failure set `is_correction=True`, retry) the Studio engine's `on_fail: retry` mirrors. Also the per-model `max_tokens` table and JSON schemas per pipeline step. | **WS1** (providers), **WS3** (retry/repair) |
| `parsing.py` | **Tolerant JSON loader** (`select_best_json_candidate`, `_loads_tolerant`, `validate_schema`) — extracts/repairs the JSON that Lofn steps emit (concepts, essence/facets, pairs) from messy model output. Reuse this for parsing step artifacts. | **WS2/WS3** (step-output parsing) |
| `o1_integration.py` | Reasoning-model wrapper (`O1ChatOpenAI`, `reasoning_level`→token budget). Historical; the Studio uses the modern Anthropic-native path (adaptive thinking) for Claude, but this shows OpenAI reasoning handling. | **WS1** |
| `image_generation.py` | Poe **image** path via `fastapi_poe` (another `fp.get_bot_response` example) + provider param shapes. NOTE: the Studio produces **prompts only** — it does NOT render media. Reference for the Poe dialect mechanics, not for wiring image gen. | **WS1** (Poe mechanics) |
| `helpers.py` | `filter_models_by_context_length`, taxonomy samplers, `truncate_prompt`. | **WS1** |
| `model_defaults.yaml` | Modality × role (`concept_medium` / `prompt`) model tiers with `Poe-` / `OR-` / bare prefixes — the heritage of FlowSpec `model_tiers`. Current model names in use. | **WS0** (FlowSpec `model_tiers`), **WS1** |
| `config.yaml` | The operator's **key names**: `ANTHROPIC_API`, `OPENAI_API`, `POE_API`, `OPEN_ROUTER_API_KEY`, `GOOGLE_API_KEY`, `LOCAL_LLM_API_BASE/KEY`. The resolver must accept these aliases in addition to the standard `*_API_KEY` names. | **WS1** (key resolution) |
| `lofjson/` | The repair library `parsing.py` depends on. | — |

**Key corrections to the plan these files force:**
- **Poe is a third dialect** (`fastapi_poe`), not `openai-compat`. See plan §7.
- **Gemini** was integrated natively (`google.genai`); an OpenAI-compat endpoint also exists — WS1 verifies and picks the simpler working path.
- **OpenRouter** is OpenAI-shaped over raw HTTP; prefer the per-request cost OpenRouter returns.

Modern-API note: for the **Anthropic** dialect, follow the current `claude-api` conventions
(adaptive thinking `{type:"adaptive"}` on 4.6+, streaming, `count_tokens`) — NOT the legacy
`ChatAnthropic(thinking={"type":"enabled","budget_tokens":25000})` seen here.
