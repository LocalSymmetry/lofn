## 2024-05-23 - Accessibility of Hidden Labels
**Learning:** Streamlit's `label_visibility="collapsed"` completely removes the label from the DOM, making it inaccessible to screen readers. `label_visibility="hidden"` should be used instead, as it keeps the label in the DOM but hides it visually, ensuring accessibility while maintaining the desired visual design.
**Action:** Always use `label_visibility="hidden"` for form elements where the label is visually redundant but necessary for accessibility.
