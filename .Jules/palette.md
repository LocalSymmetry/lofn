## 2025-12-17 - Hidden vs Collapsed Labels
**Learning:** Streamlit's `label_visibility="collapsed"` removes the label from the DOM entirely, making inputs inaccessible to screen readers. `label_visibility="hidden"` keeps the label in the DOM, preserving accessibility. **Note:** `hidden` preserves the vertical space of the label, whereas `collapsed` removes it. This trade-off (extra whitespace) is necessary for accessibility.
**Action:** Use `label_visibility="hidden"` for accessibility, but be aware of the extra vertical whitespace it introduces.
