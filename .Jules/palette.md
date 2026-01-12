# Palette's Journal

## 2025-05-15 - Streamlit Accessibility
**Learning:** In Streamlit, `label_visibility="collapsed"` completely removes the label from the DOM, making inputs inaccessible to screen readers. `label_visibility="hidden"` should be used instead, as it visually hides the label but keeps it in the DOM for assistive technology, despite the trade-off of reserving some vertical space.
**Action:** Always check `label_visibility` settings in Streamlit components and prefer "hidden" over "collapsed" unless the context is purely decorative or otherwise labeled.
