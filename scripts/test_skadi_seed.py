#!/usr/bin/env python3
"""Tests for scripts/skadi_seed.py — territory extraction, footprints, idempotence.

    python3 -m unittest discover -s scripts -p "test_skadi*.py"
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import skadi_seed as S  # noqa: E402
import validate_music_atlas as V  # noqa: E402

REAL_REPO = Path(__file__).resolve().parent.parent


class TestBaseGenreExtraction(unittest.TestCase):
    """The tricky compounds — the ones a naive strip loop gets wrong."""

    def test_single_modifier_is_stripped(self):
        self.assertEqual(S.base_genre("Industrial Frevo"), "Frevo")
        self.assertEqual(S.base_genre("Cyber Ethio-Jazz"), "Ethio-Jazz")

    def test_stacked_modifiers_are_all_stripped(self):
        self.assertEqual(S.base_genre("Avant Experimental Pop"), "Pop")
        self.assertEqual(S.base_genre("Light Old School Hip Hop"), "Old School Hip Hop")

    def test_a_modifier_word_standing_last_survives_as_the_base(self):
        # Funk, Soul, Ambient, Garage, Noise, Glitch are modifiers in first
        # position and genuine bases in last position.
        for entry, base in [("Soul Funk", "Funk"), ("Dark Ambient", "Ambient"),
                            ("Nu Soul", "Soul"), ("Cyber Noise", "Noise"),
                            ("Neo Garage", "Garage"), ("Dark Glitch", "Glitch")]:
            self.assertEqual(S.base_genre(entry), base, entry)

    def test_over_stripping_guard(self):
        # ⛔ `Psychedelic Glitch Hop` strips twice to the non-genre `Hop`. The
        # guard stops one token early. This is the whole reason NEVER_A_BASE exists.
        self.assertEqual(S.base_genre("Psychedelic Glitch Hop"), "Glitch Hop")
        self.assertEqual(S.base_genre("Dream Trip Hop"), "Trip Hop")

    def test_guard_is_minimal(self):
        # A wider guard set silently invents phantom bases: with `Jazz` in it,
        # `Ambient Jazz` never reaches the Jazz base at all.
        self.assertEqual(S.base_genre("Ambient Jazz"), "Jazz")
        self.assertEqual(S.base_genre("Nu Jazz"), "Jazz")
        self.assertEqual(S.NEVER_A_BASE, {"Hop"})

    def test_a_bare_base_is_untouched(self):
        self.assertEqual(S.base_genre("J-Pop"), "J-Pop")
        self.assertEqual(S.base_genre("West Coast Hip Hop"), "West Coast Hip Hop")


class TestTerritoryRules(unittest.TestCase):
    def test_hyphenated_compounds_reach_their_family(self):
        # A hyphen-excluding boundary sent all three of these to the unmapped list.
        self.assertEqual(S.territory_of("Anarcho-Punk"), "Punk & Hardcore")
        self.assertEqual(S.territory_of("Post-Metal"), "Metal")
        self.assertEqual(S.territory_of("P-Funk"), "Funk")

    def test_order_decides_cross_family_compounds(self):
        self.assertEqual(S.territory_of("Ska Punk"), "Ska")
        self.assertEqual(S.territory_of("Folk Metal"), "Metal")
        self.assertEqual(S.territory_of("Gospel Blues"), "Blues")
        self.assertEqual(S.territory_of("Lovers Rock"), "Reggae & Dancehall")
        self.assertEqual(S.territory_of("Dub Techno"), "Techno")

    def test_word_boundaries_do_not_leak(self):
        self.assertEqual(S.territory_of("Skool Breaks"), "UK Garage & Bassline")
        self.assertEqual(S.territory_of("Dubstep"), "Dubstep & Bass Music")
        self.assertEqual(S.territory_of("Trapstep"), "Dubstep & Bass Music")

    def test_every_base_in_the_real_corpus_maps(self):
        grouped, unmapped = S.build_territories()
        self.assertEqual(unmapped, [],
                         "a silent unsorted bucket is how a map starts lying")
        self.assertTrue(40 <= len(grouped) <= 60,
                        f"{len(grouped)} territories is outside the design's 40–60 band")
        # Assert the extraction cardinality, not just its shape.
        entries = S.read_genres()
        self.assertEqual(len(entries), 1200)
        self.assertEqual(sum(len(v) for v in grouped.values()),
                         len({S.base_genre(e) for e in entries}))


class TestFrameCategories(unittest.TestCase):
    def test_leading_blank_line_and_bom_do_not_break_the_parse(self):
        cats = S.frame_categories()
        self.assertEqual(len(cats), 55)
        self.assertEqual(cats[0], "Drums")
        self.assertNotIn("", cats)


class SeedFixture(unittest.TestCase):
    """Runs the seed against a miniature repo so nothing real is written."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        (self.root / "vault").mkdir(parents=True)
        (self.root / "output" / "songs").mkdir(parents=True)
        (self.root / "data" / "snapshots" / "suno").mkdir(parents=True)
        # Real inputs: the clustering assertions are only meaningful on the real
        # 1200-entry corpus, so those two files are copied rather than faked.
        for rel in ("vault/genres.txt", "vault/music_frames.csv", "vault/gates.yaml"):
            (self.root / rel).write_bytes((REAL_REPO / rel).read_bytes())
        (self.root / "output" / "songs" / "a.md").write_text(
            "---\ngenre_tags: \"industrial techno, dark ambient\"\n"
            "qa_flags:\n  - something unrelated\n---\n# song\n", encoding="utf-8")
        (self.root / "data" / "snapshots" / "suno" / "latest.json").write_text(
            json.dumps({"rows": [{"id": "c1", "title": "t", "tags": "gqom and amapiano"}]}),
            encoding="utf-8")
        # Stub every path the seeded claims cite, so the miniature repo can be
        # held to the SAME standard as the real one: every evidence path exists,
        # and every LISTENER-lane row cites a blind artifact.
        for claim in S.SEED_CLAIMS:
            for rel in claim[11]:
                if rel == S.NOTES:
                    continue  # the seed writes this one itself
                p = self.root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                if not p.exists():
                    p.write_text("stub", encoding="utf-8")
        self._patches = [
            mock.patch.object(S, "REPO", self.root),
            mock.patch.object(S, "ATLAS_DIR", self.root / "vault" / "music_atlas"),
            mock.patch.object(S, "SEEDS_DIR", self.root / "vault" / "music_atlas" / "seeds"),
            mock.patch.object(S, "GENRES", self.root / "vault" / "genres.txt"),
            mock.patch.object(S, "FRAMES", self.root / "vault" / "music_frames.csv"),
            mock.patch.object(S, "SONGS", self.root / "output" / "songs"),
            mock.patch.object(S, "SNAPSHOT",
                              self.root / "data" / "snapshots" / "suno" / "latest.json"),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    def seed(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            code = S.main(list(argv))
        return code, buf.getvalue()


class TestSeedIdempotence(SeedFixture):
    def test_second_run_writes_nothing(self):
        code, out = self.seed()
        self.assertEqual(code, 0)
        self.assertIn("WROTE", out)
        before = {p: p.read_bytes() for p in
                  (self.root / "vault" / "music_atlas").rglob("*.md")}
        code, out = self.seed()
        self.assertEqual(code, 0)
        self.assertNotIn("WROTE", out)
        self.assertEqual(out.count("UNCHANGED"), 6)
        after = {p: p.read_bytes() for p in
                 (self.root / "vault" / "music_atlas").rglob("*.md")}
        self.assertEqual(before, after, "the seed must be a pure function of its inputs")

    def test_a_drifted_file_is_refused_not_clobbered(self):
        self.seed()
        atlas = self.root / "vault" / "music_atlas" / "ATLAS.md"
        edited = atlas.read_text(encoding="utf-8") + "\n<!-- a human was here -->\n"
        atlas.write_text(edited, encoding="utf-8")
        code, out = self.seed()
        self.assertEqual(code, 1)
        self.assertIn("REFUSED", out)
        self.assertEqual(atlas.read_text(encoding="utf-8"), edited)
        code, _ = self.seed("--force")
        self.assertEqual(code, 0)
        self.assertNotIn("a human was here", atlas.read_text(encoding="utf-8"))

    def test_empty_extraction_is_an_error_not_a_footprint_of_zero(self):
        (self.root / "output" / "songs" / "a.md").unlink()
        with self.assertRaises(SystemExit) as ctx:
            S.footprints(S.build_territories()[0])
        self.assertIn("empty extraction", str(ctx.exception).lower())

    def test_footprint_counts_base_genres_not_territory_names(self):
        grouped, _ = S.build_territories()
        prints = S.footprints(grouped)
        # `industrial techno` in the one song; `gqom`/`amapiano` in the one clip.
        self.assertEqual(prints["Techno"], (1, 0))
        self.assertEqual(prints["Southern & East African"], (0, 1))
        # ⛔ `Vocal Harmony Traditions` must NOT match the word "vocal".
        self.assertEqual(prints["Vocal Harmony Traditions"], (0, 0))

    def test_seeded_state_passes_the_validator(self):
        self.seed()
        with mock.patch.object(V, "REPO", self.root), \
                mock.patch.object(V, "ATLAS_DIR", self.root / "vault" / "music_atlas"), \
                mock.patch.object(V, "GATES", self.root / "vault" / "gates.yaml"):
            gates, _ = V.load_gates()
            rep = V.Report()
            V.run_all(gates, rep)
        failures = {r["check"] for r in rep.rows if r["ok"] is False}
        self.assertEqual(failures, set(), rep.render())

    def test_seed_claims_never_carry_high_confidence(self):
        # Only The Scientist promotes a claim to doctrine. A CONJECTURED row
        # carries no confidence at all — it is a prediction, not an observation.
        for claim in S.SEED_CLAIMS:
            allowed = ("—",) if claim[0] == "CONJECTURED" else ("LOW", "MEDIUM")
            self.assertIn(claim[9], allowed, claim[1])

    def test_seed_claim_ns_match_the_behaviour_ledger(self):
        by_id = {c[1]: c for c in S.SEED_CLAIMS}
        self.assertEqual(by_id["SK-001"][8], 6)     # steady tempo, 6 of 6
        self.assertEqual(by_id["SK-005"][8], 2)     # interrupting silence, 2 of 2
        self.assertEqual(by_id["SK-007"][8], 4)     # foley, ledger records >= 4
        self.assertEqual(by_id["SK-012"][8], 8)     # stereo, all eight tracks
        # ⛔ THE INVERSION IS A PREDICTION, SO ITS n IS 0. It was seeded at n=2
        # carrying the 2026-08-13 run's six songs as though they were
        # observations through this instrument; that run judged the SPEC TEXT and
        # no blind pass was ever run on the renders. The paths stay in `evidence`
        # (they motivate the prediction); the COUNT does not (it would confirm
        # it). SEED_NOTES §3 says so in the file, in its own words.
        self.assertEqual(by_id["SKOP-01"][8], 0)    # the inversion, heard by nobody
        self.assertEqual(by_id["SKOP-01"][9], "—")  # and so carrying no confidence

    def test_operator_law_is_seeded_as_conjectured_not_banked(self):
        skop = next(c for c in S.SEED_CLAIMS if c[1] == "SKOP-01")
        self.assertEqual(skop[0], "CONJECTURED")
        self.assertEqual(skop[2], "OPERATOR")
        self.assertEqual(skop[5], "LISTENER")
        # The predicted verdict is unchanged and stays TRANSFORMS-scoped as the
        # panel stated it: a prediction shelf holds what we expect to find.
        self.assertEqual(skop[7], "TRANSFORMS")
        self.assertTrue(any("2026-08-13" in path for path in skop[11]),
                        "the motivating run stays cited as evidence")

    def test_seed_notes_call_the_inversion_evidence_motivating_not_confirming(self):
        # The candour is the deliverable: the row moved to n=0 and the file says
        # why, rather than the number changing quietly under the same prose.
        self.seed()
        notes = (self.root / "vault" / "music_atlas" / "seeds"
                 / "SEED_NOTES.md").read_text(encoding="utf-8")
        section = notes.split("## 3. THE 2026-08-13 INVERSION LAW")[1].split("## 4.")[0]
        self.assertIn("motivating, not confirming", section)
        self.assertIn("`n = 0`", section)
        self.assertIn("No blind listening pass", section)
        self.assertIn("replication queue", section)


if __name__ == "__main__":
    unittest.main()
