import unittest
import tempfile
from pathlib import Path
from scripts.check_brief_line_budget import (
    WORD_LABEL, band, budget_result, declarations, evaluate, word_gate,
)


class BriefBudgetReaderTests(unittest.TestCase):
    def test_yaml_serializations_preserve_the_declared_form_band(self):
        fixtures = (
            'sung_lines: [40, 72]\n',
            'sung_lines: [40, 72] # short form\n',
            'sung_lines:\n- 40\n- 72\nsung_lines_hard_floor: 40\n',
            'sung_lines:\n  - 40 # lower\n  - 72\n',
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'gates.yaml'
            for fixture in fixtures:
                with self.subTest(fixture=fixture):
                    path.write_text(fixture, encoding='utf-8')
                    limits = band(folder, str(path))
                    self.assertEqual(limits, (40, 72))
                    self.assertTrue(budget_result('8 * 7 = 56', *limits)[0])
                    self.assertFalse(budget_result('8 * 10 = 80', *limits)[0])

    def test_v6_19_word_budget_declaration_and_gate(self):
        text = 'LINE BUDGET: 20 + 6*9 = 74\nWORD BUDGET: 20*6 + 6*9*5 = 390'
        self.assertEqual(declarations(text), ['20 + 6*9 = 74'])
        self.assertEqual(declarations(text, WORD_LABEL), ['20*6 + 6*9*5 = 390'])
        with tempfile.TemporaryDirectory() as folder:
            new = Path(folder) / 'new.yaml'
            new.write_text('length_gate: words\nsung_lines: [40, 100]\nsung_words_hard_floor: 300\n'
                           'sung_words_hard_ceiling: 520  # ceiling\n', encoding='utf-8')
            mode, lo, hi = word_gate(folder, str(new))
            self.assertEqual((mode, lo, hi), ('words', 300, 520))
            self.assertTrue(budget_result('20*6 + 6*9*5 = 390', lo, hi)[0])
            self.assertFalse(budget_result('74*8 = 592', lo, hi)[0])
            self.assertFalse(budget_result('34*5 = 170', lo, hi)[0])
            # A gates file written when lines were the gate keeps that meaning.
            legacy = Path(folder) / 'legacy.yaml'
            legacy.write_text('sung_lines: [52, 76]\n', encoding='utf-8')
            self.assertEqual(word_gate(folder, str(legacy))[0], 'lines')
            # No file at all falls open to the built-in word gate.
            self.assertEqual(word_gate(folder, str(Path(folder) / 'none.yaml')), ('words', 300, 520))

    def test_plain_and_bold_declarations(self):
        self.assertEqual(declarations('LINE BUDGET: 20 + 6*9 = 74'), ['20 + 6*9 = 74'])
        self.assertEqual(declarations('**LINE BUDGET: 20 + 6*9 = 74**'), ['20 + 6*9 = 74'])
        self.assertTrue(budget_result('20 + 6*9 = 74', 70, 120)[0])

    def test_every_json_variation_is_read(self):
        text = '"line_budget": "LINE BUDGET: 8 + 64 = 72",\n"line_budget": "LINE BUDGET: 4 + 68 = 72",'
        self.assertEqual(declarations(text), ['8 + 64 = 72', '4 + 68 = 72'])
        self.assertTrue(all(budget_result(s, 70, 120)[0] for s in declarations(text)))

    def test_later_wrong_arithmetic_is_not_hidden(self):
        text = '"line_budget": "LINE BUDGET: 8 + 64 = 72",\n"line_budget": "LINE BUDGET: 4 + 68 = 80",'
        results = [budget_result(s, 70, 120)[0] for s in declarations(text)]
        self.assertEqual(results, [True, False])

    def test_context_examples_are_not_local_budgets(self):
        text = 'LINE BUDGET: <expr> = <n>\n# ===== END CREATIVE CONTEXT =====\nLINE BUDGET: 70 + 8 = 78'
        self.assertEqual(declarations(text), ['70 + 8 = 78'])

    def test_missing_bad_and_out_of_band_stay_invalid(self):
        self.assertEqual(declarations('No budget supplied'), [])
        for stated in ('8 + 4 = 80', '20 + 20 = 40', '80 + 80 = 160', '70 + 8 = 78 trailing words'):
            self.assertFalse(budget_result(stated, 70, 120)[0])
        self.assertIsNone(evaluate('__import__("os")'))


if __name__ == '__main__':
    unittest.main()
