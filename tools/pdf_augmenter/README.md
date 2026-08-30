# PDF Augmenter

A local batch converter for ML data preparation. It rasterizes every page, strips
the source document layer, applies small independent perturbations, and rebuilds an
image-only PDF at the original physical page size.

Default page treatment:

- clockwise or counter-clockwise rotation of exactly 2 or 3 degrees, with adjacent
  pages forced to use different magnitudes;
- random translation of 1-14 pixels in a continuous 0-360 degree direction;
- either Gaussian blur (radius 0.35-0.85 px) or mild sharpening (factor 1.10-1.35);
- 200 DPI rendering and JPEG quality 95.

The output contains no inherited PDF Info/XMP, annotations, attachments, searchable
text, image EXIF, or ICC profile. A valid PDF still necessarily contains structural
objects such as page dimensions and image encoding information. Filesystem timestamps
are controlled by the operating system, not embedded in the PDF.

## Windows quick start

The launcher creates a private virtual environment and installs the four local Python
dependencies on first use.

```powershell
# One file
pwsh -File tools/pdf_augmenter/run.ps1 input.pdf

# Every PDF immediately inside a directory
pwsh -File tools/pdf_augmenter/run.ps1 C:\data\pdfs -o C:\data\morphed

# Include nested directories and make the run reproducible
pwsh -File tools/pdf_augmenter/run.ps1 C:\data\pdfs -o C:\data\morphed --recursive --seed 20260824
```

Without `-o`, a file becomes `name_morphed.pdf`; a directory writes to its own
`morphed/` child. Directory structure is preserved in recursive mode. Existing
outputs are never replaced unless `--overwrite` is supplied.

## Direct Python use

```bash
python -m pip install -r tools/pdf_augmenter/requirements.txt
python tools/pdf_augmenter/pdf_augmenter.py ./pdfs -o ./morphed --recursive
```

Useful controls:

```text
--dpi 200
--rotation-min 2 --rotation-max 3
--translation-min 1 --translation-max 14
--effect random|blur|sharpen
--blur-min 0.35 --blur-max 0.85
--sharpen-min 1.10 --sharpen-max 1.35
--image-format jpeg|png
--jpeg-quality 95
--remove-footers
--footer-height-percent 10
--seed INTEGER
--overwrite
--no-manifest
```

The default JSONL manifest is external to the PDFs. Its first row records the seed
and configuration; later rows record the exact perturbation applied to each page.
The page RNG is derived from `(seed, relative input path, page number)`, so adding a
new file to a batch does not change the augmentations assigned to existing files.

For lossless image embedding, use `--image-format png`. This avoids JPEG compression
but usually creates substantially larger PDFs.

### Removing footer bands

`--remove-footers` blanks the bottom 10% of every rasterized page before rotation
and translation. This is deliberately geometric rather than OCR-based, so it also
works on scanned books. Adjust the band when necessary:

```powershell
pwsh -File tools/pdf_augmenter/run.ps1 C:\books -o C:\processed `
  -Recursive -RemoveFooters -FooterHeightPercent 9
```

Choose the smallest percentage that fully covers the footer so body text is not
removed. The exact band height is recorded per page in the external manifest.

## Safety and failure behavior

- The source PDF is never modified, even with `--overwrite`.
- A page above the default 100-megapixel render ceiling is refused before allocation.
- Password-protected or malformed PDFs are reported as failures.
- Directory runs continue after a bad file unless `--fail-fast` is used.
- Every output is reopened and checked for page count, page size, raster content,
  searchable text, annotations, forms, and metadata. `--skip-verification` disables
  this only when throughput is more important than the final integrity check.

## Tests

```powershell
python -m unittest discover -s tools/pdf_augmenter/tests -v
```
