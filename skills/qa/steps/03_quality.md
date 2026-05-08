# QA Step 3: Quality Check

## Input
- Output directory: `{output_dir}/`
- Previous QA steps: `{output_dir}/qa_step1_completeness.md`, `{output_dir}/qa_step2_contamination.md`

## Task

### Music quality checks (per song)
- [ ] Music prompt leads with emotion (first 1-3 words are emotional descriptor)
- [ ] Vocalist spec present: gender + age range + accent/ethnicity + 2 tone descriptors + 2 unique traits
- [ ] Progression map present (chronological sentences with action verbs: begins/builds/erupts/strips/fades)
- [ ] Music prompt is single paragraph (no line breaks inside prompt block)
- [ ] Music prompt 400-1000 characters
- [ ] Lyrics have varied line lengths (flag if every verse is exactly 4 lines)
- [ ] At least one `(parenthetical echo)` call-and-response
- [ ] At least one `*sound effect*` line (≤5 words)
- [ ] No use of the word "experimental" (replace with descriptive equivalent)
- [ ] Section-specific EMO: tags present
- [ ] At least one proper noun (place name, person, org, statistic) in lyrics
- [ ] SONG FORM declared with named form (not just `[SONG FORM: verse-chorus]`)
- [ ] At least 3 distinct section phase labels
- [ ] ONE decisive blow moment present
- [ ] Aftermath section present
- [ ] At least 3 different stanza lengths in the song
- [ ] At least one verifiable research fact in lyrics or music prompt

### Image quality checks (per prompt)
- [ ] Full descriptive prompt ≥ 80 words
- [ ] Subject, medium, lighting, palette, focal hierarchy named
- [ ] Emotional seed / narrative hook present
- [ ] Noun-first, present-tense (no imperative verbs opening the prompt)
- [ ] No artist names
- [ ] Concrete horizon rule: one straight edge of different material
- [ ] Legibility: primary subject reads clearly at thumbnail

### Specificity check
- At least one proper noun per output
- No generic protest clichés: "raise your voice", "we won't be silenced", "fight for what's right", "stand together"

## Save
- Quality report: `{output_dir}/qa_step3_quality.md`
