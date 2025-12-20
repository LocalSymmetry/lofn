## 2025-12-20 - Accessibility Improvements
**Learning:** Streamlit's `label_visibility="collapsed"` removes labels from the DOM, making inputs inaccessible to screen readers. `hidden` preserves them. Also, standard Sky 500 colors (#0EA5E9) often lack sufficient contrast (approx 2.7:1) against white backgrounds.
**Action:** Always use `label_visibility="hidden"` for visually hidden labels and verify color contrast ratios (aim for >4.5:1, e.g., Sky 700 #0369A1) for primary actions.
