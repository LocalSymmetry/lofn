## 2025-02-23 - Accessibility Wins
**Learning:** `label_visibility="collapsed"` completely removes the label from the DOM, making inputs inaccessible to screen readers. `label_visibility="hidden"` should be used instead to maintain accessibility while hiding the visual label. Also, standard color palettes (like Sky 500) often fail WCAG AA contrast ratios against white backgrounds; darker shades (Sky 700) are necessary for compliant text/button contrast.
**Action:** Use `label_visibility="hidden"` when hiding labels visually. Verify color contrast ratios for primary buttons and text.
