## 2024-05-23 - Accessibility in Streamlit
**Learning:** Streamlit's `label_visibility="collapsed"` completely removes the label from the DOM, making input fields inaccessible to screen readers.
**Action:** Always use `label_visibility="hidden"` instead, which preserves the label in the DOM while hiding it visually, ensuring better accessibility.
