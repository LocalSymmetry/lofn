"""Current four-song Step08 must not be rejected by a legacy six-song reader."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.validate_step import _validate, main


class MusicCardinalityTests(unittest.TestCase):
    def check_payload(self, count=4, exclude=True, **kwargs):
        items=[]
        for n in range(count):
            item={'title':f'Complete music variant {n}',
                  'music_prompt':f'Electronic composition {n}, ' + 'detailed musical material ' * 8,
                  'lyrics_prompt':f'Distinct lyric payload {n}, ' + 'complete voice and returning words ' * 8}
            if exclude: item['exclude_prompt']='Concrete excluded texture'
            items.append(item)
        if 'extra' in kwargs:
            items[0]['unknown']=kwargs.pop('extra')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'pair_01_step08_generation.md'
            p.write_text('# STEP 08\n\n## 4. Complete Step Output\n```json\n'+json.dumps({'music_prompts':items},indent=2)+'\n```\n\n## 5. Self-check\n',encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                return _validate('08',p,**kwargs)

    def test_four_with_separate_exclude_is_supported_when_declared(self):
        self.assertEqual(self.check_payload(expected_prompts=4),0)

    def test_six_legacy_triplets_keep_default(self):
        self.assertEqual(self.check_payload(count=6,exclude=False),0)

    def test_wrong_count_is_rejected(self):
        for count in (3,5,6):
            with self.subTest(count=count):
                self.assertEqual(self.check_payload(count=count,expected_prompts=4),1)

    def test_four_is_not_silently_assumed_by_legacy_call(self):
        self.assertEqual(self.check_payload(count=4),1)

    def test_unknown_field_is_rejected(self):
        self.assertEqual(self.check_payload(expected_prompts=4,extra='bad'),1)

    def test_cli_rejects_invalid_or_missing_cardinality(self):
        for argv in [['validate_step.py','--expected-prompts','5','08','file.md'],['validate_step.py','--expected-prompts']]:
            with self.subTest(argv=argv),patch('sys.argv',argv),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(),1)


class Step10JsonTests(unittest.TestCase):
    def payload(self):
        items = []
        for n in range(4):
            style = f'Electronic vocal song {n}, 116 BPM. ' + ('Rounded tones and dry drums move around the central voice. ' * 15)
            lyric = '[Theme: An unfinished offer.]\n[SONG FORM: Seven returning strophes]\n*Soft click*\n'
            for section in range(7):
                lyric += f'[Strophe {section} - EMO:Yearning - Female lead - close voice]\n'
                lyric += '\n'.join(f'Keep this little sound {n} {section} {line}' for line in range(10)) + '\n'
            items.append({'title': f'Finished Offer {n}', 'music_prompt': style,
                          'exclude_prompt': 'Continuous drum fills, ' * 22,
                          'lyrics_prompt': lyric.rstrip()})
        return {'revised_prompts': items[:2], 'synthesized_prompts': items[2:]}

    def check(self, payload):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'pair_01_step10_revision_synthesis.md'
            out = Path(d) / 'gate.json'
            p.write_text('# STEP 10 Music Song\n\n## 4. Complete Step Output\n```json\n'
                         + json.dumps(payload, indent=2) + '\n```\n\n## 5. Self-check\n', encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                result = _validate('10', p, emit_report=True, out_path=out, expected_prompts=4)
            return result, json.loads(out.read_text(encoding='utf-8')) if out.exists() else []

    def test_four_json_candidates_emit_separate_exact_measurements(self):
        payload = self.payload()
        result, rows = self.check(payload)
        self.assertEqual(result, 0)
        for check, field in [('music_prompt_chars', 'music_prompt'), ('suno_lyrics_field_max', 'lyrics_prompt'), ('suno_exclude_chars', 'exclude_prompt')]:
            measurements = [r for r in rows if r['check'] == check]
            items = payload['revised_prompts'] + payload['synthesized_prompts']
            self.assertEqual([(r['variant'], r['actual']) for r in measurements],
                             [(i, len(item[field])) for i, item in enumerate(items, 1)])

    def test_fourth_candidate_does_not_escape_existing_style_check(self):
        payload = self.payload()
        payload['synthesized_prompts'][1]['music_prompt'] = 'Too short.'
        self.assertEqual(self.check(payload)[0], 1)

    def test_two_revised_and_two_synthesized_are_required(self):
        payload = self.payload()
        payload['synthesized_prompts'].pop()
        self.assertEqual(self.check(payload)[0], 1)

    def test_missing_exclude_and_unknown_fields_are_rejected(self):
        for mutation in ('missing', 'extra'):
            payload = self.payload()
            if mutation == 'missing':
                del payload['revised_prompts'][0]['exclude_prompt']
            else:
                payload['revised_prompts'][0]['unknown'] = 'Extra data'
            with self.subTest(mutation=mutation):
                self.assertEqual(self.check(payload)[0], 1)

    def test_unknown_terminal_root_is_rejected(self):
        payload = self.payload()
        payload['extra'] = []
        self.assertEqual(self.check(payload)[0], 1)

    @staticmethod
    def lyric(lines_per_strophe, words_per_line, strophes=6, disc=False):
        lyr = '[Theme: Length test.]\n[SONG FORM: Returning strophes]\n'
        if disc:
            lyr += '[Disc_Rhythm: kick | mono]\n'
        lyr += '*Soft click*\n'
        for s in range(strophes):
            lyr += f'[Strophe {s} - EMO:Yearning - Female lead - close]\n'
            lyr += '\n'.join(' '.join(['word'] * (words_per_line - 1) + [f'{s}{i}'])
                             for i in range(lines_per_strophe)) + '\n'
        return lyr.rstrip()

    def test_v6_flags_surface_as_advisory_without_hard_fail(self):
        payload = self.payload()
        # [Disc_Rhythm:] in the box + 48 lines x 10 words = 480 sung words: over the 340-460
        # band and over 6.8 words a line, but inside the 300-520 hard gate.
        payload['revised_prompts'][0]['lyrics_prompt'] = self.lyric(8, 10, disc=True)
        result, rows = self.check(payload)
        self.assertEqual(result, 0)  # flags stay advisory (pass=None), never a hard fail
        c1_checks = {r['check']: r['pass'] for r in rows if r.get('variant') == 1}
        self.assertIn('sung_words(FLAG)', c1_checks)
        self.assertIsNone(c1_checks.get('sung_words(FLAG)'))
        self.assertIsNone(c1_checks.get('mean_words_per_line(FLAG)'))
        self.assertIsNone(c1_checks.get('disc_channel_in_lyrics_box(FLAG)'))
        self.assertTrue(c1_checks.get('sung_words'))   # V6-19: 480 words sits inside [300, 520]
        self.assertNotIn('sung_lines', c1_checks)       # the line count no longer gates
        self.assertNotIn('sung_lines(ADVISORY)', c1_checks)  # 48 lines is inside the advisory 40-100

    def test_v6_19_length_is_gated_in_sung_words_not_lines(self):
        cases = [
            # (lines per strophe, words per line, strophes, expected result, label)
            (10, 4, 10, 0, '100 lines / 400 words passes: many short lines are legal'),
            (6, 10, 7, 0, '42 lines / 420 words passes: few long lines are legal'),
            (10, 10, 6, 1, '60 lines / 600 words fails the word ceiling'),
            (12, 3, 7, 1, '84 lines / 252 words fails the word floor'),
        ]
        for per, wpl, strophes, expected, label in cases:
            with self.subTest(label=label):
                payload = self.payload()
                payload['revised_prompts'][0]['lyrics_prompt'] = self.lyric(per, wpl, strophes=strophes)
                result, rows = self.check(payload)
                self.assertEqual(result, expected)

    def test_v6_19_line_count_outside_the_advisory_range_is_a_flag_only(self):
        payload = self.payload()
        payload['revised_prompts'][0]['lyrics_prompt'] = self.lyric(6, 12, strophes=6)  # 36 lines / 432 words
        result, rows = self.check(payload)
        self.assertEqual(result, 0)
        c1_checks = {r['check']: r['pass'] for r in rows if r.get('variant') == 1}
        self.assertIn('sung_lines(ADVISORY)', c1_checks)
        self.assertIsNone(c1_checks['sung_lines(ADVISORY)'])
        self.assertTrue(c1_checks.get('sung_words'))


if __name__=='__main__':
    unittest.main()
