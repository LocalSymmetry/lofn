## 2024-05-23 - Absolute Path Robustness
**Learning:** This repo uses absolute paths (e.g., `/lofn/prompts/`) designed for Docker, which break local execution.
**Action:** Use `lofn.helpers.read_prompt` which I patched to resolve absolute paths relative to the repo root when running locally. Avoid direct `open()` calls with absolute paths.

## 2024-05-23 - Visual Hierarchy
**Learning:** Adding icons and using `type="primary"` for main action buttons significantly improves visual affordance in Streamlit apps.
**Action:** Applied Material icons (`:material/lightbulb:`, `:material/trophy:`) and primary styling to key generation buttons.
