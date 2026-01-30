## 2025-10-26 - Visual Hierarchy with Material Icons
**Learning:** Adding consistent Material Symbols to Streamlit buttons significantly improves scannability and visual hierarchy without requiring custom CSS. Using `type='primary'` sparingly for the main "happy path" action guides users effectively.
**Action:** Default to using `:material/icon_name:` for all major action buttons in Streamlit apps, reserving `type='primary'` for the single most important action in a view.
