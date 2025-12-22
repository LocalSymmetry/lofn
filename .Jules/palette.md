# Palette's Journal

## 2025-12-22 - Streamlit Accessibility Labels
**Learning:** Streamlit's `label_visibility="collapsed"` removes the label from the DOM completely, hurting screen reader accessibility. `label_visibility="hidden"` keeps it in the DOM but reserves vertical space.
**Action:** Use `hidden` over `collapsed` for input fields where visual label is redundant but accessibility is required. Accept the vertical spacing trade-off or use custom CSS to hide the reserved space if critical.
