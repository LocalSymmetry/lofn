## 2024-05-23 - Accessibility of Hidden Labels
**Learning:** Streamlit's `label_visibility="collapsed"` removes the label from the DOM entirely, making it inaccessible to screen readers. `label_visibility="hidden"` keeps it in the DOM but hides it visually.
**Action:** Always use `label_visibility="hidden"` for inputs where the design requires a hidden label, to ensure screen reader users still have context.
