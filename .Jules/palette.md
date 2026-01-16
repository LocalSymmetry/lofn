## 2025-02-18 - [Streamlit Label Visibility]
**Learning:** `label_visibility='hidden'` is preferred over `'collapsed'` in this codebase to ensure accessibility (DOM presence) even if it costs vertical space.
**Action:** Always use `hidden` for invisible labels unless specific design constraints demand `collapsed`.
