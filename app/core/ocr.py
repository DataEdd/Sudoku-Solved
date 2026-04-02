"""
Digit recognition interface and implementations.

Defines a DigitRecognizer protocol so detection and OCR are decoupled.
Swap between TesseractRecognizer (legacy) and CNNRecognizer (Phase 1).
"""

from typing import List, Protocol, Tuple

import cv2
import numpy as np


class DigitRecognizer(Protocol):
    """Protocol for digit recognition backends."""

    def predict(self, cell: np.ndarray) -> Tuple[int, float]:
        """Recognize a digit in a cell image.

        Args:
            cell: Single cell image (BGR or grayscale).

        Returns:
            (digit, confidence) where digit is 0 for empty, 1-9 otherwise.
            confidence is 0.0-1.0.
        """
        ...

    def predict_batch(self, cells: List[np.ndarray]) -> List[Tuple[int, float]]:
        """Recognize digits in multiple cells.

        Default implementation calls predict() in a loop.
        CNN implementation should override for batch inference.
        """
        ...


def preprocess_cell(cell: np.ndarray) -> np.ndarray:
    """Preprocess a cell image for digit recognition.

    Converts to grayscale, resizes to 50x50, crops margins to remove
    grid lines, applies Otsu threshold, and adds border for context.
    """
    if len(cell.shape) == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    cell = cv2.resize(cell, (50, 50))
    margin = 5
    cell = cell[margin:-margin, margin:-margin]
    _, cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cell = cv2.copyMakeBorder(cell, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
    return cell


def is_empty_cell(processed: np.ndarray, threshold: float = 0.03) -> bool:
    """Check if a preprocessed cell is empty (no digit)."""
    white_pixels = np.sum(processed == 255)
    return white_pixels / processed.size < threshold


class TesseractRecognizer:
    """Legacy OCR using Pytesseract."""

    def predict(self, cell: np.ndarray) -> Tuple[int, float]:
        import pytesseract

        processed = preprocess_cell(cell)
        if is_empty_cell(processed):
            return 0, 1.0

        config = "--psm 10 --oem 3 -c tessedit_char_whitelist=123456789"
        try:
            text = pytesseract.image_to_string(processed, config=config).strip()
            if text and text.isdigit() and 1 <= int(text) <= 9:
                return int(text), 0.7  # Tesseract doesn't give useful confidence
        except Exception:
            pass
        return 0, 0.0

    def predict_batch(self, cells: List[np.ndarray]) -> List[Tuple[int, float]]:
        return [self.predict(cell) for cell in cells]


# Re-export the real CNN recognizer now that it's implemented
from app.ml.recognizer import CNNRecognizer  # noqa: E402, F401
