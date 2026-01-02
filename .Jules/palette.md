## 2024-05-23 - [Streamlit Accessibility & Visual Hierarchy]
**Learning:** `label_visibility="collapsed"` completely removes the label from the DOM, making it inaccessible to screen readers. `label_visibility="hidden"` visually hides it but keeps it available for assistive technology.
**Action:** Always use `label_visibility="hidden"` for visually hidden labels in Streamlit.

**Learning:** Streamlit buttons can lack visual weight. Using `type="primary"` along with Material symbols (e.g., `:material/lightbulb:`) significantly improves the call-to-action visibility and user understanding.
**Action:** Use primary buttons with relevant icons for main actions in each section.
