"""
MNIST-based digit dataset for Sudoku OCR.

Downloads MNIST via torchvision and remaps labels for Sudoku use:
- MNIST digit 0 images are relabeled to class 0 (empty cell)
- MNIST digits 1-9 keep their labels
- Synthetic empty cells (blank/noisy) are added to class 0 to teach
  the model what real empty Sudoku cells look like.

Augmentations simulate real camera capture: rotation, affine warp,
noise, blur, and brightness variation.
"""

import os
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset
from torchvision import datasets, transforms

# Where to cache the downloaded MNIST data
DATA_DIR = "data/mnist"

# Known-Latin font family prefixes, matched case-insensitively against the
# basename of each discovered font file (after stripping the extension).
# Anything outside this list is rejected before we try to render digits, which
# deterministically kills LastResort.otf, Symbol.ttf, STIX math fonts,
# CJK/Myanmar/Gujarati script fonts, dingbats, and emoji fallbacks that would
# otherwise pass a naive "did any pixel render" filter and pollute
# PrintedDigitDataset with mislabeled tofu samples.
LATIN_FONT_ALLOWLIST = (
    # macOS system fonts with reliable Latin digit rendering
    "helvetica",
    "arial",
    "times",
    "georgia",
    "verdana",
    "courier",
    "menlo",
    "sfns",            # SFNSMono, SFNSRounded
    "sfcompact",
    "sfpro",
    "palatino",
    "optima",
    "futura",
    "avenir",
    "lucida",          # Lucida Grande
    "geneva",
    "trebuchet",
    "tahoma",
    "baskerville",
    "didot",
    "cochin",
    "americantypewriter",
    "andalemono",
    "monaco",
    "consolas",
    "garamond",
    "gillsans",
    "impact",
    "rockwell",
    "bodoni 72",       # narrower than "bodoni" to exclude Bodoni Ornaments (dingbats)
    "calibri",
    "cambria",
    # Common Linux distributions' default Latin fonts
    "dejavu",
    "liberation",
    "ubuntu",
)


def _is_latin_allowlisted_font(path: str) -> bool:
    """Return True if the font's basename starts with any allowlisted prefix."""
    basename = os.path.basename(path).lower()
    for ext in (".ttc", ".ttf", ".otf"):
        if basename.endswith(ext):
            basename = basename[: -len(ext)]
            break
    return any(basename.startswith(prefix) for prefix in LATIN_FONT_ALLOWLIST)


def _font_has_distinct_latin_digits(font_path: str, size: int = 20) -> bool:
    """Verify that a font renders ten distinct visible Latin digit glyphs.

    Opens the font at the given size, rasterises digits 0-9 **centred inside
    a 40×32 canvas** (wide enough to fit bold/condensed fonts like Impact or
    Arial Black without clipping — the previous 28×28 canvas with fixed
    position (4, 2) cut off wide strokes on the right, which biased the
    pairwise correlation upward for those fonts).

    Rejection criteria (either triggers a False return):

    1. **Blank glyph** — any digit renders with max pixel < 50, meaning
       the font has no visible glyph at that Unicode codepoint.
    2. **Near-duplicate pair** — any pair of digits has correlation > 0.995,
       meaning two glyphs are effectively identical.
    3. **Tofu / placeholder font** — the *mean* upper-triangular correlation
       is > 0.85. For real Latin fonts the mean is 0.36-0.77 (worst case
       Impact at 0.765) because each digit has a distinct shape; for tofu
       fonts like LastResort.otf every pair correlates near 1.0, driving
       the mean to ~1.0. Using mean instead of max is what lets bold fonts
       through: their max pairwise correlation can reach 0.97 because
       thick strokes cover more shared pixels, but the mean stays well
       under 0.85.
    """
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        return False

    tile_w, tile_h = 40, 32

    digit_flats = []
    for digit in range(10):
        img = Image.new("L", (tile_w, tile_h), 0)
        draw = ImageDraw.Draw(img)
        # Measure the actual glyph bounding box and centre it in the tile.
        bbox = draw.textbbox((0, 0), str(digit), font=font)
        glyph_w = bbox[2] - bbox[0]
        glyph_h = bbox[3] - bbox[1]
        x = (tile_w - glyph_w) // 2 - bbox[0]
        y = (tile_h - glyph_h) // 2 - bbox[1]
        draw.text((x, y), str(digit), fill=255, font=font)
        arr = np.array(img, dtype=np.float32)
        if arr.max() < 50:
            return False  # blank glyph — font doesn't support this digit
        digit_flats.append(arr.flatten())

    mat = np.stack(digit_flats)
    if (mat.std(axis=1) < 1e-6).any():
        return False

    corr = np.corrcoef(mat)
    if np.isnan(corr).any():
        return False

    upper_tri = np.triu(np.ones((10, 10), dtype=bool), k=1)
    off_diag = corr[upper_tri]

    if float(off_diag.max()) > 0.995:
        return False  # near-duplicate pair
    if float(off_diag.mean()) > 0.85:
        return False  # overall tofu-like
    return True


class EmptyCellDataset(Dataset):
    """Synthetic empty Sudoku cells grounded in measured GT statistics.

    Since 2026-04-11 (v4.2), ``EmptyCellDataset`` is the **sole** source of
    class 0 samples in the training pool. MNIST handwritten '0' glyphs used
    to contribute ~5923 mislabeled samples; those were dropped via
    ``_load_mnist_no_zero`` because they're round digit shapes, not blank
    newspaper cells. PrintedDigitDataset and Chars74KFontDataset both
    already skip the digit 0 in their render loops.

    **Target distribution — GT empty cells (pre-normalize, post-invert):**

        mean   p25=38   p50=56   p75=78
        std    p25=3    p50=8    p75=18
        p5     p25=29   p50=44   p75=70
        p95    p25=55   p50=82   p75=105
        lap_var p25=20  p50=97   p75=425

    (Measured on 1358 real empty cells from 38 GT newspaper photos via
    ``notebooks/gt_cell_measurement.py``; the frozen stats live at
    ``notebooks/gt_cell_stats.json:gt_empty``.)

    Pre-v4.2 the four variants all had mean ~5, p5=0, p95=16 — ~50 points
    darker than GT. Synthetic samples were modelling "ideal pure-white
    paper" (= near-black in MNIST polarity), not real off-white newsprint
    paper. Post-v4.2 the four variants target the GT distribution above.
    All four are in MNIST polarity (white-ish digit / bright features on
    dark-ish paper background); random bright features represent what
    survives of grid lines or faint ink residues after the production
    invert + normalize pipeline.
    """

    def __init__(self, count: int = 5000, size: int = 28, seed: int = 42):
        self.size = size
        self.count = count
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self._generate(idx)
        tensor = torch.from_numpy(img).unsqueeze(0).float() / 255.0
        return tensor, 0

    def _smooth_noise(self, std: float, s: int) -> np.ndarray:
        """Generate spatially-correlated Gaussian noise matching paper texture.

        Pixel-independent ``rng.normal`` noise has very high spatial frequency
        (every pixel is an independent sample), so ``cv2.Laplacian(...).var()``
        on the resulting image is ~400-500 for std 5 — far above the measured
        GT empty-cell laplacian variance of ~97. Real paper texture is
        spatially correlated at the 2-3 px scale; simulating that needs a
        smoothed noise field. We generate white noise at the target size,
        blur it with a small Gaussian (σ ≈ 1.2 px — enough to correlate
        neighbours without losing all structure), then rescale to the
        target std so the distribution's magnitude matches regardless of
        the blur's energy loss.
        """
        import cv2
        noise = self.rng.normal(0, 1, (s, s)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=1.2)
        cur_std = float(noise.std())
        if cur_std > 1e-6:
            noise = noise * (std / cur_std)
        return noise

    def _generate(self, idx: int) -> np.ndarray:
        """Generate one empty-cell sample matching GT pre-normalize stats.

        Four variants, selected by ``idx % 4``:
          0. **Clean paper** — uniform base + spatially-correlated noise
          1. **Lighting gradient** — base + shallow spatial gradient
          2. **Grid-line remnant** — base + a brighter line at one edge
             (a grid line that survived the inner-10% margin crop and
             became bright after MNIST-polarity invert)
          3. **Faint ink residue** — base + a small brighter blob (a cell
             that has been partially erased or bleed-through from a
             neighbour)

        All four use ``_smooth_noise`` for the texture component rather
        than pixel-independent Gaussian noise, matching the GT laplacian
        variance of ~97 (pre-pipeline) rather than the ~400+ produced by
        white noise.
        """
        s = self.size
        choice = idx % 4

        # Paper base intensity — matches GT empty cell mean (~56) with
        # natural variation across the p25-p75 range (38-78).
        base = int(self.rng.randint(40, 75))

        if choice == 0:
            # Clean paper: uniform base + spatially-smoothed paper noise.
            # Wider base range + wider noise std range than v4.2 pre-audit
            # so two clean-paper samples with similar seeds don't end up
            # near-identical (the variation check flagged 31% near-duplicate
            # pairs when base was 40-75 and noise std was 3-6 fixed).
            wide_base = float(base) + float(self.rng.uniform(-8, 8))
            img = np.full((s, s), wide_base, dtype=np.float32)
            img += self._smooth_noise(self.rng.uniform(3.0, 10.0), s)
            return np.clip(img, 0, 255).astype(np.uint8)

        if choice == 1:
            # Lighting gradient: paper base + spatial gradient across one axis
            # (widened span range for more diversity), plus texture noise
            # with variable strength.
            span = int(self.rng.randint(-25, 26))
            grad = np.linspace(base, base + span, s, dtype=np.float32)
            if self.rng.random() < 0.5:
                img = np.tile(grad, (s, 1)).astype(np.float32)
            else:
                img = np.tile(grad.reshape(-1, 1), (1, s)).astype(np.float32)
            img += self._smooth_noise(self.rng.uniform(2.0, 5.0), s)
            return np.clip(img, 0, 255).astype(np.uint8)

        if choice == 2:
            # Grid-line remnant: paper base with a brighter edge line.
            # After the original white-paper-on-black-grid cell gets
            # inverted to MNIST polarity, leftover grid-line pixels near
            # the border appear as brighter ink-like strokes. Width 1-3
            # px, intensity ~70-140 (p75-p95 range of GT empty cells).
            img = np.full((s, s), base, dtype=np.float32)
            img += self._smooth_noise(3.5, s)
            line_intensity = int(self.rng.randint(70, 140))
            line_width = int(self.rng.randint(1, 4))
            edge = int(self.rng.randint(4))
            if edge == 0:
                img[:line_width, :] = line_intensity
            elif edge == 1:
                img[-line_width:, :] = line_intensity
            elif edge == 2:
                img[:, :line_width] = line_intensity
            else:
                img[:, -line_width:] = line_intensity
            return np.clip(img, 0, 255).astype(np.uint8)

        # choice == 3: paper with a faint ink residue
        # Small brighter blob placed at one of the four tile corners/edges,
        # never at the tile centre. A centred blob would look like a digit
        # "0"/"O" to the variation-check heuristic and structurally to the
        # CNN — we explicitly want class 0 to never contain centred round
        # clusters. The blob sits at a randomised near-corner so the
        # centroid is always > 8 px from (14, 14).
        img = np.full((s, s), base, dtype=np.float32)
        img += self._smooth_noise(3.5, s)
        # Four corner quadrants; pick one and place the blob inside it.
        quadrant = int(self.rng.randint(4))
        cx_min, cx_max = (3, 10) if quadrant in (0, 2) else (18, 25)
        cy_min, cy_max = (3, 10) if quadrant in (0, 1) else (18, 25)
        cx = int(self.rng.randint(cx_min, cx_max))
        cy = int(self.rng.randint(cy_min, cy_max))
        r = int(self.rng.randint(2, 5))  # slightly smaller than before
        yy, xx = np.ogrid[:s, :s]
        blob_mask = (yy - cy) ** 2 + (xx - cx) ** 2 < r * r
        blob_intensity = int(self.rng.randint(70, 110))
        img[blob_mask] = blob_intensity
        return np.clip(img, 0, 255).astype(np.uint8)


class PrintedDigitDataset(Dataset):
    """Synthetic printed digits rendered from 67 validated system fonts.

    Generates digit images (1-9) in various macOS/Linux Latin fonts at 28×28,
    white-on-black, matching the MNIST format. Covers the printed digit
    domain that MNIST (handwritten only) misses.

    Class 0 is not generated here — empty cells are handled by EmptyCellDataset.

    **Rendering (post-2026-04-11 v4.1 fix):** digits are rendered on a 56×56
    intermediate canvas at font sizes 32-48 (2× the final output), all in-class
    augmentation (±10° rotation, additive Gaussian noise, 3×3 Gaussian blur)
    is applied on the 56×56 canvas, and the final step is a ``cv2.resize`` to
    28×28 via ``INTER_AREA``. The big-canvas approach eliminates a clipping
    bug present in all pre-v4.1 checkpoints: rendering at the final 28×28
    size on wide/bold fonts (Impact, Arial Black, Bodoni 72, Rockwell, Avenir
    Next Condensed) or after rotation would push glyph ink past the tile
    boundary, producing training samples with partial digits. The v4 attempt
    to drop this dataset entirely (removing all 67 macOS-specific fonts) was
    reverted after measuring a −5.6 filled-cell regression on real photos;
    Chars74K's 812-font English-Font archive does NOT subsume the macOS
    system fonts for classes 2/4/6. See
    ``memory/project_v4_drop_printed_2026_04_11.md`` for the v3 → v4 → v4.1
    measurement trail.
    """

    def __init__(
        self,
        count_per_digit: int = 500,
        size: int = 28,
        seed: int = 42,
    ):
        self.size = size
        self.rng = np.random.RandomState(seed)

        # Discover system fonts
        font_dirs = [
            "/System/Library/Fonts",
            "/System/Library/Fonts/Supplemental",
            "/Library/Fonts",
            "/usr/share/fonts",
            "/usr/share/fonts/truetype",
        ]
        font_paths = []
        for d in font_dirs:
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.endswith((".ttf", ".ttc", ".otf")):
                    font_paths.append(os.path.join(d, f))

        # Stage 1: keep only basenames that match a known Latin font family.
        allowlisted = [fp for fp in font_paths if _is_latin_allowlisted_font(fp)]

        # Stage 2: verify each survivor actually renders ten distinct Latin
        # digits (catches any tofu-rendering font the allow-list missed).
        self.fonts = [
            fp for fp in allowlisted if _font_has_distinct_latin_digits(fp)
        ]

        if len(self.fonts) < 10:
            raise RuntimeError(
                f"PrintedDigitDataset: only {len(self.fonts)} fonts passed the "
                f"Latin-digit filter (need at least 10). Searched "
                f"{len(font_paths)} font files across {font_dirs}; "
                f"{len(allowlisted)} matched the allow-list; "
                f"{len(self.fonts)} passed the distinct-digit signature test. "
                f"Either install more Latin fonts or extend "
                f"LATIN_FONT_ALLOWLIST in app/ml/dataset.py."
            )

        self.images = []
        self.labels = []

        for digit in range(1, 10):
            for _ in range(count_per_digit):
                img = self._render_digit(digit)
                self.images.append(img)
                self.labels.append(digit)

    def _render_digit(self, digit: int) -> np.ndarray:
        """Render a single digit sample via the 2026-04-11 v4.1 big-canvas pattern.

        Pipeline:
          1. Render the glyph onto a 56×56 PIL canvas at font size 32-48
             (2× the target 28×28 final size).
          2. Centre via textbbox measurement + bbox offset correction + ±4 px
             jitter (2× the old ±2 px to match the 2× canvas).
          3. Apply in-class augmentation (±10° rotation, additive noise, 3×3
             blur) on the 56×56 canvas so rotation/affine steps have 4× the
             pixel headroom before clipping.
          4. ``cv2.resize`` to 28×28 via ``INTER_AREA`` (the standard for
             downsizing; anti-aliases by pixel-area averaging).

        The pre-v4.1 implementation rendered directly on 28×28 at font sizes
        16-24 with ±2 px jitter, which clipped wide/bold fonts after rotation
        or when the downstream ``AugmentedDataset`` geometric chain applied
        its ±15° rotation + (0.85, 1.15) scale + 8% translate. The v4.1
        big-canvas approach eliminates that bug without changing the overall
        digit-with-margin distribution the model sees at inference.
        """
        from PIL import Image, ImageDraw, ImageFont
        import cv2

        s = self.size            # 28 — final output size
        big_s = s * 2            # 56 — render canvas (2× headroom for augmentation)

        img = Image.new("L", (big_s, big_s), 0)
        draw = ImageDraw.Draw(img)

        # Random font and size. self.fonts is guaranteed non-empty and
        # every entry is a validated Latin-digit-rendering font.
        fp = self.fonts[self.rng.randint(len(self.fonts))]
        font_size = self.rng.randint(32, 48)  # 2× the old 16-24 range
        font = ImageFont.truetype(fp, font_size)

        text = str(digit)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # Centre the ink inside the big canvas. ``textbbox`` returns pen-start
        # coordinates, not ink-start coordinates, so we subtract ``bbox[0]``
        # and ``bbox[1]`` to position the ink box's top-left at the intended
        # centred location. Jitter is ±4 px on the 56×56 canvas, matching the
        # old ±2 px on the 28×28 canvas after the downsize.
        x = (big_s - tw) // 2 - bbox[0] + int(self.rng.randint(-4, 5))
        y = (big_s - th) // 2 - bbox[1] + int(self.rng.randint(-4, 5))
        draw.text((x, y), text, fill=255, font=font)

        arr = np.array(img)

        # Random augmentation — all on the 56×56 canvas so rotation corners
        # don't clip glyph ink.
        if self.rng.random() < 0.3:
            angle = self.rng.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((big_s / 2, big_s / 2), angle, 1.0)
            arr = cv2.warpAffine(arr, M, (big_s, big_s))

        if self.rng.random() < 0.2:
            noise = self.rng.normal(0, 8, arr.shape)
            arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if self.rng.random() < 0.2:
            arr = cv2.GaussianBlur(arr, (3, 3), 0)

        # Final step: downsize 56×56 → 28×28 via INTER_AREA. This preserves
        # the "digit sits inside the tile with whitespace margin" distribution
        # that matches what the inference ``CNNRecognizer._preprocess`` chain
        # produces after its inner-10% margin crop + resize.
        arr = cv2.resize(arr, (s, s), interpolation=cv2.INTER_AREA)

        return arr

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx]
        tensor = torch.from_numpy(img).unsqueeze(0).float() / 255.0
        return tensor, self.labels[idx]


class Chars74KFontDataset(Dataset):
    """Chars74K English-Font digit subset with font-disjoint train/test split.

    Downloads (once, cached) the Chars74K English-Font archive, extracts the
    digit classes (Sample001=0, Sample002=1, ..., Sample010=9), and partitions
    the ~1016 per-class font renderings by instance index so the same font
    never appears in both train and test splits. This lets the test split
    measure generalization to *unseen fonts*, not font memorization.

    Follows the PrintedDigitDataset convention: digit 0 (Sample001) is
    excluded because class 0 in this project represents "empty cell"
    (see MNIST relabeling note at the top of this module), and we do not
    want printed-zero glyphs polluting that class.

    Images are cropped to their bounding box, padded to square, resized to
    28×28, and inverted to MNIST polarity (white digit on black background).
    """

    URL = "https://info-ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz"
    ARCHIVE_NAME = "EnglishFnt.tgz"
    CACHE_DIR = "data/chars74k"

    def __init__(
        self,
        split: str = "train",
        size: int = 28,
        train_frac: float = 0.8,
        seed: int = 42,
        download: bool = True,
    ):
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        from pathlib import Path
        import cv2

        self.split = split
        self.size = size

        cache = Path(self.CACHE_DIR)
        archive_path = cache / self.ARCHIVE_NAME
        extract_root = cache / "English" / "Fnt"

        if download and not extract_root.exists():
            self._download_and_extract(cache, archive_path)

        if not extract_root.exists():
            raise RuntimeError(
                f"Chars74KFontDataset: extracted data not found at "
                f"{extract_root}. Pass download=True or manually download "
                f"{self.URL} to {archive_path} and extract."
            )

        # Digit 0 is excluded (see docstring). Iterate Sample002..Sample010
        # which correspond to digits 1..9 in Chars74K's labeling.
        all_files = []
        for digit in range(1, 10):
            sample_idx = digit + 1  # Sample002 -> digit 1, ..., Sample010 -> digit 9
            sample_dir = extract_root / f"Sample{sample_idx:03d}"
            if not sample_dir.is_dir():
                raise RuntimeError(
                    f"Chars74KFontDataset: expected directory {sample_dir} "
                    f"missing after extraction — archive structure may have "
                    f"changed."
                )
            for f in sorted(sample_dir.iterdir()):
                if f.suffix.lower() != ".png":
                    continue
                # Filename pattern: img{class:03d}-{instance:05d}.png
                parts = f.stem.rsplit("-", 1)
                if len(parts) != 2 or not parts[1].isdigit():
                    continue
                instance = int(parts[1])
                all_files.append((f, digit, instance))

        if not all_files:
            raise RuntimeError(
                f"Chars74KFontDataset: no PNG files found under {extract_root}"
            )

        # Font-disjoint split: shuffle all unique instance indices with a
        # fixed seed and partition by train_frac. Same instance index across
        # different Sample directories is the same font identity.
        all_instances = sorted({instance for _, _, instance in all_files})
        rng = np.random.RandomState(seed)
        shuffled = all_instances.copy()
        rng.shuffle(shuffled)
        split_point = int(len(shuffled) * train_frac)
        train_ids = set(shuffled[:split_point])
        test_ids = set(shuffled[split_point:])

        selected = train_ids if split == "train" else test_ids
        self.font_ids = sorted(selected)

        # Filter files to the selected split and load/preprocess
        self.images: list[np.ndarray] = []
        self.labels: list[int] = []
        for fpath, digit, instance in all_files:
            if instance not in selected:
                continue
            img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            processed = self._preprocess(img)
            if processed is None:
                continue
            self.images.append(processed)
            self.labels.append(digit)

        if not self.images:
            raise RuntimeError(
                f"Chars74KFontDataset: {split} split produced zero samples "
                f"after preprocessing"
            )

    def _preprocess(self, img: np.ndarray) -> "np.ndarray | None":
        """Crop to bounding box, pad to square, resize, invert polarity."""
        import cv2

        s = self.size
        # Chars74K-Fnt images are black ink on white paper. Invert to find
        # the ink region via simple thresholding.
        _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
        ys, xs = np.where(binary > 0)
        if len(ys) == 0:
            return None  # blank image — skip

        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        crop = img[y0:y1, x0:x1]

        # Pad to square with white (paper) background
        h, w = crop.shape
        side = max(h, w)
        pad_top = (side - h) // 2
        pad_bottom = side - h - pad_top
        pad_left = (side - w) // 2
        pad_right = side - w - pad_left
        square = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=255,
        )

        # Resize then invert to MNIST polarity (white digit on black)
        resized = cv2.resize(square, (s, s), interpolation=cv2.INTER_AREA)
        return (255 - resized).astype(np.uint8)

    def _download_and_extract(self, cache: "Path", archive_path: "Path") -> None:
        import tarfile
        import urllib.request

        cache.mkdir(parents=True, exist_ok=True)

        if not archive_path.exists():
            print(f"Chars74KFontDataset: downloading {self.URL} ...")
            urllib.request.urlretrieve(self.URL, archive_path)
            print(f"  saved to {archive_path}")

        print(f"Chars74KFontDataset: extracting {archive_path} ...")
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(cache)
        print(f"  extracted to {cache}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx]
        tensor = torch.from_numpy(img).unsqueeze(0).float() / 255.0
        return tensor, self.labels[idx]


class AugmentedDataset(Dataset):
    """Wraps a dataset and applies training-time augmentation.

    When ``augment=True`` the pipeline is:

        1. ``_apply_noise``  — legacy stochastic noise/brightness/blur
                               (pre-existing behaviour, unchanged)
        2. geometric         — random rotation ±15° + affine translate/scale/shear
        3. ``_apply_newsprint`` — blur + contrast compression + brightness pedestal,
                                  with parameters drawn from ranges that were
                                  empirically tuned against the 38 GT newspaper
                                  cells (see ``notebooks/gt_cell_measurement.py``
                                  and ``notebooks/gt_cell_stats.json``) so that
                                  the *distribution* of augmented training
                                  samples matches the distribution the CNN
                                  sees at inference on real photos.

    When ``augment=False`` the sample passes through unchanged (used for val
    and test splits so they remain reproducible and comparable to historical
    checkpoints).
    """

    def __init__(self, base: Dataset, augment: bool = True):
        self.base = base
        self.augment = augment

        if augment:
            # PIL-in, PIL-out pipeline — final ToTensor happens in __getitem__
            # so we can interleave numpy/cv2 operations around it.
            self.geometric = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomRotation(15, fill=0),
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(0.08, 0.08),
                        scale=(0.85, 1.15),
                        shear=(-10, 10),
                        fill=0,
                    ),
                ]
            )
        else:
            self.geometric = None

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.base[idx]

        # Normalize to uint8 numpy in (H, W) MNIST polarity.
        if isinstance(img, torch.Tensor):
            np_img = (img.squeeze(0).numpy() * 255.0).astype(np.uint8)
        else:
            pil = img.convert("L") if getattr(img, "mode", "L") != "L" else img
            np_img = np.asarray(pil, dtype=np.uint8)

        if self.augment:
            np_img = self._apply_noise(np_img)
            np_img = np.asarray(self.geometric(np_img), dtype=np.uint8)
            np_img = self._apply_newsprint(np_img)

        tensor = torch.from_numpy(np_img).unsqueeze(0).float() / 255.0
        return tensor, label

    def _apply_noise(self, img: np.ndarray) -> np.ndarray:
        if random.random() < 0.4:
            noise = np.random.normal(0, random.uniform(5, 15), img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if random.random() < 0.3:
            shift = random.randint(-20, 20)
            img = np.clip(img.astype(np.int16) + shift, 0, 255).astype(np.uint8)
        if random.random() < 0.25:
            import cv2

            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        return img

    def _apply_newsprint(self, img: np.ndarray) -> np.ndarray:
        """Match the empirical GT newspaper-cell distribution seen at inference.

        Target statistics (median over 1720 filled GT cells from 38 photos,
        **measured through the exact CNNRecognizer._preprocess chain** —
        margin crop + resize + invert + min-max normalize to [0, 255]):

            mean=50, p5=9, p95=189, p95-p5=177, laplacian_var=2374

        The inference preprocessing min-max-normalizes every cell, so this
        augmentation mirrors the full preprocessing chain: Gaussian blur
        → min-max normalize → **pedestal shift**. The sigma range was
        tuned against the target laplacian variance (2374) via a grid
        sweep.

        **2026-04-11 (v4):** added the post-normalize pedestal step.
        Pre-v4 measurement showed train p5=0 vs GT p5=9 — training samples
        had pure-black backgrounds after min-max, while real newspaper
        cells retain a ~9-level pedestal from paper tone. The pedestal
        range [5, 15] is grounded in the GT postnorm p5 distribution
        (p25=7, p50=9, p75=13 from notebooks/gt_cell_stats.json under
        gt_filled_postnorm.p5). BatchNorm was previously absorbing the
        gap but the model had also learnt the shortcut "p5=0 → empty
        cell", which contributed to empty-cell hallucinations on real
        photos (memory/lesson_newsprint_match_partial.md).

        Measured at sigma ∈ [0.5, 0.8] pre-pedestal (v3):
            mean=34, p5=0, p95=218, p95-p5=218, lap_var=2594

        Measured post-pedestal (v4, expected — verify post-retrain):
            mean≈44, p5≈10, p95≈228, p95-p5≈218, lap_var≈2594

        The pedestal adds an additive offset but does not change the
        high-frequency content, so laplacian variance is unaffected.
        """
        import cv2

        sigma = random.uniform(0.5, 0.8)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        # L2 pedestal shift was tested on 2026-04-11 and reverted —
        # it caused a real-photo regression alongside the drop-PrintedDigit
        # change. Bisection kept drop-PrintedDigit, reverted L2. See
        # memory/project_v4_drop_printed_2026_04_11.md for the numbers.
        return img


def _load_mnist(train: bool) -> Dataset:
    """Download and load MNIST, returning tensors in (1, 28, 28) float [0,1]."""
    return datasets.MNIST(
        root=DATA_DIR,
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )


def _load_mnist_no_zero(train: bool) -> Dataset:
    """Load MNIST, then drop every sample labelled 0.

    **Why:** class 0 in this project means "empty cell", NOT the digit zero.
    MNIST's ~5923 handwritten '0' glyphs in the train split (and ~980 in the
    test split) are round digit-shaped images that semantically belong to
    *no* Sudoku class — Sudoku cells only ever contain blanks or digits 1-9.
    Including them in class 0 pollutes the empty-cell label with high-ink
    round shapes, teaching the model that "class 0 = (blank) OR (a round
    digit)". That's a semantic contradiction with the production inference
    contract, which uses the class-0 probability only as an empty-vs-filled
    gate.

    This helper returns a ``torch.utils.data.Subset`` that excludes every
    label-0 sample. ``PrintedDigitDataset`` and ``Chars74KFontDataset``
    already skip the digit-0 glyph by construction (their inner loops
    iterate ``range(1, 10)``); the filter here closes the last remaining
    source of "digit-zero" samples that were leaking into class 0.
    """
    base = datasets.MNIST(
        root=DATA_DIR,
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )
    targets = base.targets
    if hasattr(targets, "numpy"):
        targets = targets.numpy()
    else:
        targets = np.asarray(targets)
    keep_indices = [int(i) for i in np.where(targets != 0)[0]]
    return torch.utils.data.Subset(base, keep_indices)


def create_datasets(
    empty_cell_count: int = 5000,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Create train/val/test datasets for the SudokuCNN digit recogniser.

    Composition (post-2026-04-11 v4.2):

      Train:
        - MNIST train MINUS label 0 (~54k) — handwritten digits 1-9. The
          ~5923 MNIST '0' glyphs are dropped because they're round digit
          shapes, not blank cells — class 0 in this project means "empty
          cell", not "digit zero". See _load_mnist_no_zero docstring.
        - EmptyCellDataset (~5k) — synthetic empty cell variants, rewritten
          in v4.2 to match the GT pre-normalize distribution (mean ~56,
          std ~8, paper pedestal p5 ~44). Source of ALL class 0 samples.
        - PrintedDigitDataset (~4.5k) — macOS/Linux system-font printed
          digits 1-9 from 67 validated Latin fonts (see LATIN_FONT_ALLOWLIST)
        - Chars74KFontDataset(split="train") — ~7.3k printed digits 1-9
          rendered from ~800 unique computer fonts (font-disjoint from test)

      Val:
        - 10% holdout from the concatenated train pool (same composition)

      Test:
        - MNIST test MINUS label 0 (~9k) — same rationale as train
        - EmptyCellDataset test split (1k) — the sole class 0 test source
        - Chars74KFontDataset(split="test") — ~1.8k printed digits 1-9
          from ~200 fonts DISJOINT from the train split.

    **2026-04-11 history (v4 and v4.1):**

    v4 (dropped, not shipped) removed PrintedDigitDataset from training
    because the 28×28 render canvas was clipping wide/bold font glyphs.
    The retrain caused a measured −5.6 filled-cell regression on real
    photos (66.6% → 61.0%) — Chars74K's 812-font archive did NOT subsume
    the 67 macOS-specific bold/condensed fonts (Impact, Arial Black,
    Bodoni 72, Rockwell, Avenir Next Condensed, SFNS variants). Classes
    2, 4, and 6 were hit hardest, each losing ~13 percentage points.

    v4.1 fixes the clipping in place: ``PrintedDigitDataset._render_digit``
    now renders on a 56×56 canvas at font sizes 32-48, applies the
    in-class rotation / noise / blur augmentation on the big canvas, and
    resizes to 28×28 via INTER_AREA at the end. This eliminates the
    affine-induced clipping while preserving the "digit-with-margin"
    distribution that matches the inference pipeline's post-margin-crop
    output. See ``memory/project_v4_drop_printed_2026_04_11.md`` for the
    full v3 → v4 → v4.1 measurement trail.

    The train split is wrapped with augmentation that matches the empirical
    distribution of the 38 GT newspaper cells (blur + min-max normalize);
    see AugmentedDataset._apply_newsprint for details. Val and test pass
    through unaugmented so they remain reproducible and directly comparable
    to historical checkpoint numbers; the **augmented** eval distribution
    is measured separately in train.py by re-wrapping ``val_ds.base`` and
    ``test_ds.base`` with ``AugmentedDataset(augment=True)`` (L1 from the
    2026-04-11 pipeline review).

    The 38 GT newspaper photos are intentionally NOT in any of these splits
    — they remain held-out real-photo evaluation via evaluate_ocr.py, per
    the data-leakage rule.
    """
    rng = torch.Generator().manual_seed(seed)

    # Synthetic sources that go into BOTH train and val (split later).
    # MNIST is filtered to drop label 0 (see _load_mnist_no_zero).
    mnist_train = _load_mnist_no_zero(train=True)
    empty_train = EmptyCellDataset(count=empty_cell_count, seed=seed)
    printed_train = PrintedDigitDataset(count_per_digit=500, seed=seed)
    chars74k_train = Chars74KFontDataset(split="train", seed=seed)

    full_train = ConcatDataset(
        [mnist_train, empty_train, printed_train, chars74k_train]
    )

    # 90/10 train/val split over the concatenated pool
    n = len(full_train)
    val_size = n // 10
    train_size = n - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        full_train, [train_size, val_size], generator=rng
    )

    train_ds = AugmentedDataset(train_subset, augment=True)
    val_ds = AugmentedDataset(val_subset, augment=False)

    # Test split: MNIST test MINUS label 0 + held-out empty cells + font-
    # disjoint Chars74K test split. The "first version of the dataset that
    # actually measures generalisation to unseen printed-digit fonts" is
    # preserved; the label-0 filter only removes handwritten-0 glyphs from
    # the test MNIST half, which aren't a meaningful test signal anyway
    # (the production pipeline never predicts class 0 directly — it uses
    # the class-0 probability as an empty-vs-filled gate).
    mnist_test = _load_mnist_no_zero(train=False)
    empty_test = EmptyCellDataset(count=1000, seed=seed + 1)
    chars74k_test = Chars74KFontDataset(split="test", seed=seed)
    test_ds = AugmentedDataset(
        ConcatDataset([mnist_test, empty_test, chars74k_test]),
        augment=False,
    )

    return train_ds, val_ds, test_ds
