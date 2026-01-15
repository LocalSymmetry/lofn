## 2025-10-26 - Streamlit Button Hierarchy & A11y
**Learning:** `label_visibility="collapsed"` removes the label from the accessibility tree, harming screen reader users. `label_visibility="hidden"` is the correct choice for visually hiding labels while maintaining accessibility.
**Action:** Always use `hidden` instead of `collapsed` for inputs unless the label is redundant with another accessible element.

**Learning:** Streamlit's `type="primary"` combined with `icon` (v1.34+) significantly improves visual hierarchy for main actions compared to default buttons.
**Action:** Audit main CTAs to use `type="primary"` and relevant Material icons.
