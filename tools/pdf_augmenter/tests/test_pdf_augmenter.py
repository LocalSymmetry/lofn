from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from pypdf import PdfReader
from reportlab.pdfgen import canvas


MODULE_PATH = Path(__file__).resolve().parents[1] / "pdf_augmenter.py"
SPEC = importlib.util.spec_from_file_location("pdf_augmenter", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
pdf_augmenter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pdf_augmenter
SPEC.loader.exec_module(pdf_augmenter)


def make_source_pdf(path: Path) -> None:
    document = canvas.Canvas(str(path), pagesize=(300, 400))
    document.setAuthor("Sensitive Author")
    document.setTitle("Sensitive Title")
    document.drawString(36, 360, "Searchable source text - page one")
    document.drawCentredString(150, 12, "CONFIDENTIAL FOOTER - 1")
    document.showPage()
    document.setPageSize((500, 250))
    document.drawString(36, 210, "Searchable source text - page two")
    document.drawCentredString(250, 12, "CONFIDENTIAL FOOTER - 2")
    document.save()


class PdfAugmenterTests(unittest.TestCase):
    def test_parameter_sampling_is_deterministic_and_within_bounds(self) -> None:
        config = pdf_augmenter.AugmentationConfig(
            rotation_degrees=pdf_augmenter.NumberRange(2.0, 3.0),
            translation_pixels=pdf_augmenter.NumberRange(1.0, 14.0),
        )
        first = pdf_augmenter.sample_page_parameters(
            pdf_augmenter._stable_page_rng(123, "a.pdf", 0), config
        )
        second = pdf_augmenter.sample_page_parameters(
            pdf_augmenter._stable_page_rng(123, "a.pdf", 0), config
        )
        self.assertEqual(first, second)
        self.assertIn(abs(first.rotation_degrees), {2.0, 3.0})
        self.assertGreaterEqual(first.translation_pixels, 1.0)
        self.assertLessEqual(first.translation_pixels, 14.0)
        self.assertAlmostEqual(
            (first.dx_pixels**2 + first.dy_pixels**2) ** 0.5,
            first.translation_pixels,
        )

    def test_augmentation_keeps_canvas_size_and_strips_image_metadata(self) -> None:
        image = Image.new("RGB", (120, 80), "white")
        image.info["private"] = "remove me"
        config = pdf_augmenter.AugmentationConfig(effect="blur")
        parameters = pdf_augmenter.PageParameters(
            rotation_degrees=3.0,
            translation_pixels=5.0,
            direction_degrees=90.0,
            dx_pixels=0.0,
            dy_pixels=5.0,
            effect="blur",
            effect_strength=0.5,
        )
        output = pdf_augmenter.augment_image(image, parameters, config)
        self.assertEqual(output.size, image.size)
        self.assertEqual(output.mode, "RGB")
        self.assertEqual(output.info, {})

    def test_footer_band_is_blank_before_augmentation(self) -> None:
        image = Image.new("RGB", (100, 100), "white")
        image.paste("black", (0, 90, 100, 100))
        config = pdf_augmenter.AugmentationConfig(
            rotation_degrees=pdf_augmenter.NumberRange(0.0, 0.0),
            translation_pixels=pdf_augmenter.NumberRange(0.0, 0.0),
            sharpen_factor=pdf_augmenter.NumberRange(1.0, 1.0),
            effect="sharpen",
            remove_footers=True,
            footer_height_percent=10.0,
        )
        parameters = pdf_augmenter.PageParameters(
            rotation_degrees=0.0,
            translation_pixels=0.0,
            direction_degrees=0.0,
            dx_pixels=0.0,
            dy_pixels=0.0,
            effect="sharpen",
            effect_strength=1.0,
        )
        output = pdf_augmenter.augment_image(image, parameters, config)
        self.assertEqual(
            output.crop((0, 90, 100, 100)).getextrema(),
            ((255, 255), (255, 255), (255, 255)),
        )

    def test_conversion_rasterizes_scrubs_and_preserves_mixed_page_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.pdf"
            output = root / "source_morphed.pdf"
            make_source_pdf(source)

            config = pdf_augmenter.AugmentationConfig(dpi=96, image_format="png")
            records = pdf_augmenter.convert_pdf(
                source,
                output,
                config,
                master_seed=77,
                input_key="source.pdf",
                verify=True,
            )

            self.assertEqual(len(records), 2)
            self.assertEqual(
                {abs(float(record["rotation_degrees"])) for record in records},
                {2.0, 3.0},
            )
            reader = PdfReader(output)
            self.assertFalse(reader.metadata)
            self.assertNotIn("/Metadata", reader.trailer["/Root"])
            self.assertEqual(len(reader.pages), 2)
            expected_sizes = ((300.0, 400.0), (500.0, 250.0))
            for page, expected_size in zip(reader.pages, expected_sizes):
                self.assertEqual((float(page.mediabox.width), float(page.mediabox.height)), expected_size)
                self.assertEqual((page.extract_text() or "").strip(), "")
                self.assertNotIn("/Annots", page)
                self.assertGreaterEqual(len(page.images), 1)

    def test_recursive_jobs_preserve_relative_directories_and_ignore_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            nested = root / "nested"
            nested.mkdir()
            make_source_pdf(root / "one.pdf")
            make_source_pdf(nested / "two.PDF")
            output_root = root / "morphed"
            output_root.mkdir()
            make_source_pdf(output_root / "old.pdf")
            make_source_pdf(root / "already_morphed.pdf")

            jobs, destination = pdf_augmenter.build_jobs(
                root, output_root, recursive=True, suffix="_morphed"
            )

            self.assertEqual(destination, output_root)
            self.assertEqual([job.input_key for job in jobs], ["nested/two.PDF", "one.pdf"])
            self.assertEqual(
                [job.output_path.relative_to(output_root).as_posix() for job in jobs],
                ["nested/two_morphed.pdf", "one_morphed.pdf"],
            )

    def test_manifest_is_jsonl_and_carries_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = Path(temporary_directory) / "manifest.jsonl"
            config = pdf_augmenter.AugmentationConfig()
            pdf_augmenter._write_manifest(
                manifest,
                master_seed=42,
                config=config,
                page_records=[{"record_type": "page", "page": 1}],
                failures=[],
                overwrite=False,
            )
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["seed"], 42)
            self.assertEqual(rows[1]["page"], 1)


if __name__ == "__main__":
    unittest.main()
