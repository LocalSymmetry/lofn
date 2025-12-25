## 2025-05-23 - [Streamlit Label Accessibility]
**Learning:** Streamlit's `label_visibility='collapsed'` completely removes the label from the DOM, making the input inaccessible to screen readers.
**Action:** Always use `label_visibility='hidden'` instead, which keeps the label in the DOM but hides it visually, ensuring accessibility while maintaining the desired visual design.
