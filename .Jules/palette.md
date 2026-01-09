## 2024-05-22 - Improved Streamlit Accessibility
**Learning:** `label_visibility="collapsed"` completely removes the label from the accessibility tree, making it invisible to screen readers. `label_visibility="hidden"` is the correct choice as it visually hides the label but keeps it accessible in the DOM.
**Action:** Always use `label_visibility="hidden"` when a visual label is redundant but context is needed for screen readers.
