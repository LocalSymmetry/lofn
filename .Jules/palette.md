## 2024-10-26 - [Standardizing Primary Actions in Streamlit]
**Learning:** In Streamlit interfaces with dense sidebars and multiple inputs, the primary "Generate" action often gets lost. Using `type="primary"` alongside consistent Material icons (e.g., `:material/lightbulb:` for creation, `:material/trophy:` for competition) creates a necessary visual anchor.
**Action:** For every major interaction block, identify the single primary action and apply the `type="primary"` style with a semantic icon. Ensure consistent icon usage for similar actions across different tabs (Music/Video/Image).
