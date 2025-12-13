## 2024-05-22 - [Accessible Name Mismatch]
**Learning:** Using `label="Short Name"` with `label_visibility="collapsed"` alongside a visible `st.subheader("Long Descriptive Name")` creates a mismatch between what sighted users see and what screen readers announce. This confuses voice control users who say the visible text but the system expects the hidden label.
**Action:** Ensure the hidden `label` text matches the visible header text (e.g., set `label="Describe Your Idea"` if the header is "Describe Your Idea").
