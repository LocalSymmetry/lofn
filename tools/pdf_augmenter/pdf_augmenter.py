#!/usr/bin/env python3
"""Rasterize PDFs and apply small, reproducible page-level augmentations."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import secrets
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pypdfium2 as pdfium
from PIL import Image, ImageEnhance, ImageFilter
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


@dataclass(frozen=True)
class NumberRange:
    low: float
    high: float

    def validate(self, label: str, *, minimum: float = 0.0) -> None:
        if self.low < minimum or self.high < minimum:
            raise ValueError(f"{label} values must be at least {minimum}")
        if self.low > self.high:
            raise ValueError(f"{label} minimum cannot exceed its maximum")

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.low, self.high)


@dataclass(frozen=True)
class AugmentationConfig:
    dpi: int = 200
    rotation_degrees: NumberRange = NumberRange(2.0, 3.0)
    translation_pixels: NumberRange = NumberRange(1.0, 14.0)
    blur_radius: NumberRange = NumberRange(0.35, 0.85)
    sharpen_factor: NumberRange = NumberRange(1.10, 1.35)
    effect: str = "random"
    image_format: str = "jpeg"
    jpeg_quality: int = 95
    max_megapixels: float = 100.0
    remove_footers: bool = False
    footer_height_percent: float = 10.0
    background_rgb: tuple[int, int, int] = (255, 255, 255)

    def validate(self) -> None:
        if not 36 <= self.dpi <= 1200:
            raise ValueError("dpi must be between 36 and 1200")
        self.rotation_degrees.validate("rotation")
        self.translation_pixels.validate("translation")
        self.blur_radius.validate("blur radius")
        self.sharpen_factor.validate("sharpen factor", minimum=1.0)
        if self.effect not in {"random", "blur", "sharpen"}:
            raise ValueError("effect must be random, blur, or sharpen")
        if self.image_format not in {"jpeg", "png"}:
            raise ValueError("image format must be jpeg or png")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg quality must be between 1 and 100")
        if self.max_megapixels <= 0:
            raise ValueError("max megapixels must be positive")
        if not 0.1 <= self.footer_height_percent <= 40.0:
            raise ValueError("footer height percent must be between 0.1 and 40")


@dataclass(frozen=True)
class PageParameters:
    rotation_degrees: float
    translation_pixels: float
    direction_degrees: float
    dx_pixels: float
    dy_pixels: float
    effect: str
    effect_strength: float


@dataclass(frozen=True)
class ConversionJob:
    input_path: Path
    output_path: Path
    input_key: str


def _stable_page_rng(master_seed: int, input_key: str, page_index: int) -> random.Random:
    material = f"{master_seed}\0{input_key}\0{page_index}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def sample_page_parameters(
    rng: random.Random,
    config: AugmentationConfig,
    *,
    previous_rotation_magnitude: float | None = None,
) -> PageParameters:
    rotation_range = config.rotation_degrees
    if float(rotation_range.low).is_integer() and float(rotation_range.high).is_integer():
        candidates = [
            float(value)
            for value in range(int(rotation_range.low), int(rotation_range.high) + 1)
        ]
        if previous_rotation_magnitude is not None and len(candidates) > 1:
            candidates = [
                value
                for value in candidates
                if not math.isclose(value, previous_rotation_magnitude)
            ]
        magnitude = rng.choice(candidates)
    else:
        magnitude = rotation_range.sample(rng)
    rotation = magnitude * rng.choice((-1.0, 1.0))
    distance = config.translation_pixels.sample(rng)
    direction = rng.uniform(0.0, 360.0)
    direction_radians = math.radians(direction)
    dx = distance * math.cos(direction_radians)
    dy = distance * math.sin(direction_radians)
    effect = rng.choice(("blur", "sharpen")) if config.effect == "random" else config.effect
    strength = (
        config.blur_radius.sample(rng)
        if effect == "blur"
        else config.sharpen_factor.sample(rng)
    )
    return PageParameters(
        rotation_degrees=rotation,
        translation_pixels=distance,
        direction_degrees=direction,
        dx_pixels=dx,
        dy_pixels=dy,
        effect=effect,
        effect_strength=strength,
    )


def _flatten_to_rgb(image: Image.Image, background: tuple[int, int, int]) -> Image.Image:
    converted = image.convert("RGBA")
    flattened = Image.new("RGB", converted.size, background)
    flattened.paste(converted, mask=converted.getchannel("A"))
    flattened.info.clear()
    return flattened


def augment_image(
    image: Image.Image,
    parameters: PageParameters,
    config: AugmentationConfig,
) -> Image.Image:
    clean = _flatten_to_rgb(image, config.background_rgb)
    if config.remove_footers:
        footer_height = max(
            1, round(clean.height * config.footer_height_percent / 100.0)
        )
        clean.paste(
            config.background_rgb,
            (0, clean.height - footer_height, clean.width, clean.height),
        )
    transformed = clean.rotate(
        parameters.rotation_degrees,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        translate=(parameters.dx_pixels, parameters.dy_pixels),
        fillcolor=config.background_rgb,
    )
    if parameters.effect == "blur":
        transformed = transformed.filter(
            ImageFilter.GaussianBlur(radius=parameters.effect_strength)
        )
    else:
        transformed = ImageEnhance.Sharpness(transformed).enhance(
            parameters.effect_strength
        )
    transformed.info.clear()
    return transformed


def _encode_image(image: Image.Image, config: AugmentationConfig) -> bytes:
    stream = io.BytesIO()
    if config.image_format == "jpeg":
        image.save(
            stream,
            format="JPEG",
            quality=config.jpeg_quality,
            subsampling=0,
            optimize=True,
            progressive=False,
            exif=b"",
        )
    else:
        image.save(stream, format="PNG", optimize=True)
    return stream.getvalue()


def _temporary_path(directory: Path, stem: str, label: str) -> Path:
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{stem}.", suffix=f".{label}.pdf", dir=directory, delete=False
    )
    handle.close()
    return Path(handle.name)


def _scrub_pdf_metadata(source: Path, destination: Path) -> None:
    reader = PdfReader(source)
    writer = PdfWriter()
    for source_page in reader.pages:
        source_page.pop(NameObject("/Metadata"), None)
        source_page.pop(NameObject("/PieceInfo"), None)
        writer.add_page(source_page)
    writer.metadata = None
    writer.root_object.pop(NameObject("/Metadata"), None)
    writer.root_object.pop(NameObject("/PieceInfo"), None)
    with destination.open("wb") as output_stream:
        writer.write(output_stream)


def verify_output_pdf(
    output_path: Path,
    expected_page_sizes: Sequence[tuple[float, float]],
) -> None:
    reader = PdfReader(output_path)
    if len(reader.pages) != len(expected_page_sizes):
        raise RuntimeError(
            f"verification failed: expected {len(expected_page_sizes)} pages, "
            f"found {len(reader.pages)}"
        )

    metadata = reader.metadata
    nonempty_metadata = {
        str(key): value
        for key, value in (metadata or {}).items()
        if value not in (None, "")
    }
    if nonempty_metadata:
        raise RuntimeError(
            f"verification failed: PDF metadata remains: {sorted(nonempty_metadata)}"
        )

    root = reader.trailer["/Root"]
    forbidden_root_keys = {"/Metadata", "/AcroForm", "/OpenAction"}
    present_root_keys = forbidden_root_keys.intersection(root.keys())
    if present_root_keys:
        raise RuntimeError(
            "verification failed: interactive or metadata objects remain: "
            f"{sorted(present_root_keys)}"
        )

    for page_index, (page, expected_size) in enumerate(
        zip(reader.pages, expected_page_sizes), start=1
    ):
        if "/Metadata" in page or "/Annots" in page:
            raise RuntimeError(
                f"verification failed: page {page_index} contains metadata or annotations"
            )
        extracted_text = page.extract_text() or ""
        if extracted_text.strip():
            raise RuntimeError(
                f"verification failed: page {page_index} still has searchable text"
            )
        if len(page.images) < 1:
            raise RuntimeError(
                f"verification failed: page {page_index} has no raster image"
            )
        actual_size = (float(page.mediabox.width), float(page.mediabox.height))
        if any(abs(actual - expected) > 0.75 for actual, expected in zip(actual_size, expected_size)):
            raise RuntimeError(
                f"verification failed: page {page_index} size changed from "
                f"{expected_size} points to {actual_size} points"
            )


def convert_pdf(
    input_path: Path,
    output_path: Path,
    config: AugmentationConfig,
    *,
    master_seed: int,
    input_key: str | None = None,
    overwrite: bool = False,
    verify: bool = True,
) -> list[dict[str, object]]:
    config.validate()
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if input_path == output_path:
        raise ValueError("refusing to overwrite the source PDF")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered_temp = _temporary_path(output_path.parent, output_path.stem, "render")
    scrubbed_temp = _temporary_path(output_path.parent, output_path.stem, "scrub")
    page_records: list[dict[str, object]] = []
    expected_page_sizes: list[tuple[float, float]] = []
    stable_input_key = input_key or input_path.name

    try:
        document = pdfium.PdfDocument(str(input_path))
        try:
            if len(document) == 0:
                raise ValueError(f"input PDF has no pages: {input_path}")
            pdf_canvas = canvas.Canvas(
                str(rendered_temp),
                pagesize=(1.0, 1.0),
                pageCompression=1,
                invariant=1,
            )
            try:
                previous_rotation_magnitude: float | None = None
                for page_index in range(len(document)):
                    page = document[page_index]
                    try:
                        width_points, height_points = page.get_size()
                        scale = config.dpi / 72.0
                        width_pixels = max(1, math.ceil(width_points * scale))
                        height_pixels = max(1, math.ceil(height_points * scale))
                        megapixels = width_pixels * height_pixels / 1_000_000.0
                        if megapixels > config.max_megapixels:
                            raise ValueError(
                                f"page {page_index + 1} would render at {megapixels:.1f} MP, "
                                f"above the {config.max_megapixels:.1f} MP safety limit"
                            )

                        bitmap = page.render(scale=scale)
                        try:
                            rendered_image = bitmap.to_pil().copy()
                        finally:
                            bitmap.close()

                        rng = _stable_page_rng(master_seed, stable_input_key, page_index)
                        parameters = sample_page_parameters(
                            rng,
                            config,
                            previous_rotation_magnitude=previous_rotation_magnitude,
                        )
                        previous_rotation_magnitude = abs(
                            parameters.rotation_degrees
                        )
                        augmented_image = augment_image(
                            rendered_image, parameters, config
                        )
                        encoded_image = _encode_image(augmented_image, config)

                        page_size = (float(width_points), float(height_points))
                        expected_page_sizes.append(page_size)
                        pdf_canvas.setPageSize(page_size)
                        pdf_canvas.drawImage(
                            ImageReader(io.BytesIO(encoded_image)),
                            0,
                            0,
                            width=width_points,
                            height=height_points,
                            preserveAspectRatio=False,
                            mask=None,
                        )
                        pdf_canvas.showPage()
                        page_records.append(
                            {
                                "record_type": "page",
                                "input": stable_input_key,
                                "output": output_path.name,
                                "page": page_index + 1,
                                "page_width_points": width_points,
                                "page_height_points": height_points,
                                "render_width_pixels": augmented_image.width,
                                "render_height_pixels": augmented_image.height,
                                "footer_removed": config.remove_footers,
                                "footer_height_pixels": (
                                    max(
                                        1,
                                        round(
                                            augmented_image.height
                                            * config.footer_height_percent
                                            / 100.0
                                        ),
                                    )
                                    if config.remove_footers
                                    else 0
                                ),
                                **asdict(parameters),
                            }
                        )
                    finally:
                        page.close()
            finally:
                pdf_canvas.save()
        finally:
            document.close()

        _scrub_pdf_metadata(rendered_temp, scrubbed_temp)
        if verify:
            verify_output_pdf(scrubbed_temp, expected_page_sizes)
        os.replace(scrubbed_temp, output_path)
        return page_records
    finally:
        rendered_temp.unlink(missing_ok=True)
        scrubbed_temp.unlink(missing_ok=True)


def _is_inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def build_jobs(
    input_path: Path,
    output_path: Path | None,
    *,
    recursive: bool,
    suffix: str,
) -> tuple[list[ConversionJob], Path]:
    input_path = input_path.resolve()
    if not suffix or any(separator in suffix for separator in ("/", "\\")):
        raise ValueError("suffix must be a non-empty filename fragment")

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"input file is not a PDF: {input_path}")
        if output_path is None:
            destination = input_path.with_name(f"{input_path.stem}{suffix}.pdf")
        else:
            candidate = output_path.resolve()
            destination = (
                candidate / f"{input_path.stem}{suffix}.pdf"
                if candidate.exists() and candidate.is_dir()
                else candidate
            )
            if destination.suffix.lower() != ".pdf":
                destination = destination / f"{input_path.stem}{suffix}.pdf"
        return [ConversionJob(input_path, destination, input_path.name)], destination.parent

    if not input_path.is_dir():
        raise FileNotFoundError(f"input path does not exist: {input_path}")

    destination_root = (
        output_path.resolve() if output_path is not None else input_path / "morphed"
    )
    if destination_root.exists() and destination_root.is_file():
        raise ValueError("directory input requires an output directory")

    candidates: Iterable[Path] = input_path.rglob("*") if recursive else input_path.iterdir()
    jobs: list[ConversionJob] = []
    for candidate in sorted(candidates, key=lambda item: str(item).lower()):
        if not candidate.is_file() or candidate.suffix.lower() != ".pdf":
            continue
        if _is_inside(candidate, destination_root):
            continue
        if candidate.stem.endswith(suffix):
            continue
        relative = candidate.relative_to(input_path)
        destination_relative = relative.with_name(f"{relative.stem}{suffix}.pdf")
        jobs.append(
            ConversionJob(
                input_path=candidate,
                output_path=destination_root / destination_relative,
                input_key=relative.as_posix(),
            )
        )
    if not jobs:
        raise FileNotFoundError(f"no PDF files found under {input_path}")
    return jobs, destination_root


def _config_to_dict(config: AugmentationConfig) -> dict[str, object]:
    result = asdict(config)
    result["background_rgb"] = list(config.background_rgb)
    return result


def _write_manifest(
    manifest_path: Path,
    *,
    master_seed: int,
    config: AugmentationConfig,
    page_records: Sequence[dict[str, object]],
    failures: Sequence[dict[str, str]],
    overwrite: bool,
) -> None:
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"manifest already exists: {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(f".{manifest_path.name}.{secrets.token_hex(6)}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(
                json.dumps(
                    {
                        "record_type": "run",
                        "seed": master_seed,
                        "config": _config_to_dict(config),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            for record in page_records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            for failure in failures:
                stream.write(
                    json.dumps(
                        {"record_type": "failure", **failure}, sort_keys=True
                    )
                    + "\n"
                )
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rasterize one PDF or every PDF in a directory, then independently "
            "rotate, translate, and mildly blur or sharpen each page."
        )
    )
    parser.add_argument("input", type=Path, help="input PDF or directory")
    parser.add_argument("-o", "--output", type=Path, help="output PDF or directory")
    parser.add_argument("--recursive", action="store_true", help="scan subdirectories")
    parser.add_argument("--seed", type=int, help="master seed for reproducible output")
    parser.add_argument("--dpi", type=int, default=200, help="rasterization DPI (default: 200)")
    parser.add_argument("--rotation-min", type=float, default=2.0)
    parser.add_argument("--rotation-max", type=float, default=3.0)
    parser.add_argument("--translation-min", type=float, default=1.0)
    parser.add_argument("--translation-max", type=float, default=14.0)
    parser.add_argument("--blur-min", type=float, default=0.35)
    parser.add_argument("--blur-max", type=float, default=0.85)
    parser.add_argument("--sharpen-min", type=float, default=1.10)
    parser.add_argument("--sharpen-max", type=float, default=1.35)
    parser.add_argument(
        "--effect", choices=("random", "blur", "sharpen"), default="random"
    )
    parser.add_argument(
        "--image-format", choices=("jpeg", "png"), default="jpeg"
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-megapixels", type=float, default=100.0)
    parser.add_argument(
        "--remove-footers",
        action="store_true",
        help="blank the bottom footer band before page augmentation",
    )
    parser.add_argument(
        "--footer-height-percent",
        type=float,
        default=10.0,
        help="percentage of page height blanked by --remove-footers (default: 10)",
    )
    parser.add_argument("--suffix", default="_morphed")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="skip the post-write metadata, text, image, and page-size checks",
    )
    manifest_group = parser.add_mutually_exclusive_group()
    manifest_group.add_argument("--manifest", type=Path, help="JSONL manifest path")
    manifest_group.add_argument(
        "--no-manifest", action="store_true", help="do not write a JSONL manifest"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    master_seed = args.seed if args.seed is not None else secrets.randbits(64)
    if master_seed < 0:
        print("error: seed must be non-negative", file=sys.stderr)
        return 2

    config = AugmentationConfig(
        dpi=args.dpi,
        rotation_degrees=NumberRange(args.rotation_min, args.rotation_max),
        translation_pixels=NumberRange(args.translation_min, args.translation_max),
        blur_radius=NumberRange(args.blur_min, args.blur_max),
        sharpen_factor=NumberRange(args.sharpen_min, args.sharpen_max),
        effect=args.effect,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
        max_megapixels=args.max_megapixels,
        remove_footers=args.remove_footers,
        footer_height_percent=args.footer_height_percent,
    )

    try:
        config.validate()
        jobs, destination_root = build_jobs(
            args.input,
            args.output,
            recursive=args.recursive,
            suffix=args.suffix,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.manifest is not None:
        manifest_path = args.manifest.resolve()
    elif args.no_manifest:
        manifest_path = None
    elif args.input.resolve().is_file():
        manifest_path = jobs[0].output_path.with_suffix(".manifest.jsonl")
    else:
        manifest_path = destination_root / "manifest.jsonl"

    if manifest_path is not None and manifest_path.exists() and not args.overwrite:
        print(
            f"error: manifest already exists: {manifest_path} (use --overwrite)",
            file=sys.stderr,
        )
        return 2

    print(f"Seed: {master_seed}")
    page_records: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    converted_files = 0

    for index, job in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {job.input_path} -> {job.output_path}")
        try:
            records = convert_pdf(
                job.input_path,
                job.output_path,
                config,
                master_seed=master_seed,
                input_key=job.input_key,
                overwrite=args.overwrite,
                verify=not args.skip_verification,
            )
            page_records.extend(records)
            converted_files += 1
            print(f"  wrote {len(records)} page(s)")
        except Exception as error:  # keep a directory batch moving
            failure = {"input": job.input_key, "error": str(error)}
            failures.append(failure)
            print(f"  FAILED: {error}", file=sys.stderr)
            if args.fail_fast:
                break

    if manifest_path is not None:
        try:
            _write_manifest(
                manifest_path,
                master_seed=master_seed,
                config=config,
                page_records=page_records,
                failures=failures,
                overwrite=args.overwrite,
            )
            print(f"Manifest: {manifest_path}")
        except Exception as error:
            print(f"error: could not write manifest: {error}", file=sys.stderr)
            return 1

    print(
        f"Complete: {converted_files}/{len(jobs)} PDF(s), "
        f"{len(page_records)} page(s), {len(failures)} failure(s)."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
