## 2024-05-23 - Delightful Loading & Accessible Labels
**Learning:** Streamlit's `st.spinner` supports dynamic text, which can turn waiting time into a moment of brand personality. Also, `st.text_area` with `label_visibility="collapsed"` still exposes the label to screen readers, so ensuring it matches the visual header avoids confusion.
**Action:** Use dynamic, context-aware loading messages instead of static "Generating..." text. Always sync hidden labels with visual headings.
