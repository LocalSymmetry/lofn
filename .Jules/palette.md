## 2025-05-15 - Streamlit Button Styles Overridden
**Learning:** `lofn/style.css` uses `!important` on `div.stButton > button`, overriding Streamlit's native `type="primary"` visual distinction.
**Action:** Relied on `icon` and `help` attributes to distinguish primary actions instead of color alone.
