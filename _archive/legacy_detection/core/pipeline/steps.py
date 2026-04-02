"""
Preprocessing step implementations.

Each step is a self-contained transformation that can be composed
into pipelines. Steps use OpenCV internally but expose a clean interface.
"""

from typing import Any, Dict

import cv2
import numpy as np

from .base import PreprocessingStep


class GrayscaleStep(PreprocessingStep):
    """
    Convert BGR image to grayscale.

    Formula: G = 0.299R + 0.587G + 0.114B

    These weights reflect human eye sensitivity:
    - Green: Most sensitive (0.587)
    - Red: Medium (0.299)
    - Blue: Least sensitive (0.114)
    """

    name = "grayscale"
    description = "Convert to grayscale: G = 0.299R + 0.587G + 0.114B"

    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        if len(image.shape) == 2:
            return image  # Already grayscale
        if image.shape[2] == 4:
            # BGRA to grayscale
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class GaussianBlurStep(PreprocessingStep):
    """
    Apply Gaussian blur to reduce noise.

    Gaussian kernel formula:
    K_sigma(x, y) = (1 / 2*pi*sigma^2) * exp(-(x^2 + y^2) / 2*sigma^2)

    Benefits:
    - Reduces high-frequency noise
    - Smooths out small imperfections
    - Prepares image for edge detection
    """

    name = "gaussian_blur"
    description = "Gaussian blur for noise reduction"

    def process(
        self,
        image: np.ndarray,
        kernel_size: int = 5,
        sigma: float = 0
    ) -> np.ndarray:
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    def get_default_params(self) -> Dict[str, Any]:
        return {"kernel_size": 5, "sigma": 0}

    def validate_params(self, params: Dict[str, Any]) -> None:
        kernel_size = params.get("kernel_size", 5)
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")


class AdaptiveThresholdStep(PreprocessingStep):
    """
    Apply adaptive thresholding to create binary image.

    For each pixel (i, j):
    - Calculate local mean mu in a block_size x block_size window
    - T_{i,j} = 255 if image_{i,j} < mu_{i,j} - C else 0

    Handles uneven lighting, shadows, different paper colors.
    """

    name = "adaptive_threshold"
    description = "Adaptive threshold for varying lighting"

    def process(
        self,
        image: np.ndarray,
        block_size: int = 11,
        c: int = 2,
        method: str = "gaussian"
    ) -> np.ndarray:
        adapt_method = (
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            if method == "gaussian"
            else cv2.ADAPTIVE_THRESH_MEAN_C
        )
        return cv2.adaptiveThreshold(
            image, 255, adapt_method, cv2.THRESH_BINARY_INV, block_size, c
        )

    def get_default_params(self) -> Dict[str, Any]:
        return {"block_size": 11, "c": 2, "method": "gaussian"}

    def validate_params(self, params: Dict[str, Any]) -> None:
        block_size = params.get("block_size", 11)
        if block_size % 2 == 0:
            raise ValueError(f"block_size must be odd, got {block_size}")
        if block_size < 3:
            raise ValueError(f"block_size must be >= 3, got {block_size}")


class BinaryThresholdStep(PreprocessingStep):
    """
    Apply simple binary thresholding.

    Pixels above threshold become 255, below become 0.
    Can use Otsu's method to auto-determine threshold.
    """

    name = "binary_threshold"
    description = "Simple binary threshold"

    def process(
        self,
        image: np.ndarray,
        threshold: int = 127,
        use_otsu: bool = False,
        invert: bool = False
    ) -> np.ndarray:
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        if use_otsu:
            thresh_type |= cv2.THRESH_OTSU
            threshold = 0  # Otsu determines this
        _, result = cv2.threshold(image, threshold, 255, thresh_type)
        return result

    def get_default_params(self) -> Dict[str, Any]:
        return {"threshold": 127, "use_otsu": False, "invert": False}


class SobelGradientStep(PreprocessingStep):
    """
    Compute Sobel gradient magnitude.

    Sobel kernels detect edges by computing image gradients.
    Magnitude = sqrt(Gx^2 + Gy^2)
    """

    name = "sobel"
    description = "Sobel gradient magnitude"

    def process(
        self,
        image: np.ndarray,
        ksize: int = 3,
        normalize: bool = True
    ) -> np.ndarray:
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
        magnitude = np.sqrt(gx**2 + gy**2)

        if normalize and magnitude.max() > 0:
            magnitude = (magnitude / magnitude.max() * 255)

        return magnitude.astype(np.uint8)

    def get_default_params(self) -> Dict[str, Any]:
        return {"ksize": 3, "normalize": True}


class CannyEdgeStep(PreprocessingStep):
    """
    Canny edge detection with hysteresis.

    Algorithm steps:
    1. Compute gradient magnitude and direction (Sobel)
    2. Non-maximum suppression (thin edges to 1 pixel)
    3. Hysteresis thresholding:
       - Strong edges: gradient > high_threshold (kept)
       - Weak edges: low_threshold < gradient < high_threshold
       - Weak edges kept only if connected to strong edges
    """

    name = "canny"
    description = "Canny edge detection with hysteresis"

    def process(
        self,
        image: np.ndarray,
        low_threshold: int = 50,
        high_threshold: int = 150,
        aperture_size: int = 3
    ) -> np.ndarray:
        return cv2.Canny(
            image, low_threshold, high_threshold, apertureSize=aperture_size
        )

    def get_default_params(self) -> Dict[str, Any]:
        return {"low_threshold": 50, "high_threshold": 150, "aperture_size": 3}

    def validate_params(self, params: Dict[str, Any]) -> None:
        low = params.get("low_threshold", 50)
        high = params.get("high_threshold", 150)
        if low >= high:
            raise ValueError(f"low_threshold ({low}) must be < high_threshold ({high})")


class MorphologyStep(PreprocessingStep):
    """
    Morphological operations: dilate, erode, open, close.

    - Dilate: Expands white regions (connects broken lines)
    - Erode: Shrinks white regions (removes small noise)
    - Open: Erode then dilate (removes small objects)
    - Close: Dilate then erode (fills small holes)
    """

    name = "morphology"
    description = "Morphological operations (dilate/erode/open/close)"

    def process(
        self,
        image: np.ndarray,
        operation: str = "dilate",
        kernel_size: int = 3,
        iterations: int = 1,
        kernel_shape: str = "rect"
    ) -> np.ndarray:
        # Create kernel
        if kernel_shape == "rect":
            shape = cv2.MORPH_RECT
        elif kernel_shape == "ellipse":
            shape = cv2.MORPH_ELLIPSE
        elif kernel_shape == "cross":
            shape = cv2.MORPH_CROSS
        else:
            shape = cv2.MORPH_RECT

        kernel = cv2.getStructuringElement(shape, (kernel_size, kernel_size))

        # Apply operation
        if operation == "dilate":
            return cv2.dilate(image, kernel, iterations=iterations)
        elif operation == "erode":
            return cv2.erode(image, kernel, iterations=iterations)
        elif operation == "open":
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == "close":
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "operation": "dilate",
            "kernel_size": 3,
            "iterations": 1,
            "kernel_shape": "rect"
        }


class MedianBlurStep(PreprocessingStep):
    """
    Apply median blur for salt-and-pepper noise removal.

    Replaces each pixel with the median of its neighborhood.
    Excellent for removing impulse noise while preserving edges.
    """

    name = "median_blur"
    description = "Median blur for impulse noise removal"

    def process(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        return cv2.medianBlur(image, kernel_size)

    def get_default_params(self) -> Dict[str, Any]:
        return {"kernel_size": 5}

    def validate_params(self, params: Dict[str, Any]) -> None:
        kernel_size = params.get("kernel_size", 5)
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")


class BilateralFilterStep(PreprocessingStep):
    """
    Apply bilateral filter for edge-preserving smoothing.

    Considers both spatial distance and intensity difference,
    smoothing while keeping edges sharp.
    """

    name = "bilateral"
    description = "Bilateral filter (edge-preserving smoothing)"

    def process(
        self,
        image: np.ndarray,
        d: int = 9,
        sigma_color: float = 75,
        sigma_space: float = 75
    ) -> np.ndarray:
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    def get_default_params(self) -> Dict[str, Any]:
        return {"d": 9, "sigma_color": 75, "sigma_space": 75}


class ContrastStep(PreprocessingStep):
    """
    Adjust image contrast using histogram equalization or CLAHE.

    CLAHE (Contrast Limited Adaptive Histogram Equalization)
    works on small tiles to handle varying lighting.
    """

    name = "contrast"
    description = "Contrast enhancement (histogram equalization or CLAHE)"

    def process(
        self,
        image: np.ndarray,
        method: str = "clahe",
        clip_limit: float = 2.0,
        tile_size: int = 8
    ) -> np.ndarray:
        if method == "histogram":
            return cv2.equalizeHist(image)
        elif method == "clahe":
            clahe = cv2.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=(tile_size, tile_size)
            )
            return clahe.apply(image)
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_default_params(self) -> Dict[str, Any]:
        return {"method": "clahe", "clip_limit": 2.0, "tile_size": 8}


class InvertStep(PreprocessingStep):
    """
    Invert image (bitwise NOT).

    Converts white to black and vice versa.
    Useful when grid lines need to be white for detection.
    """

    name = "invert"
    description = "Invert image (bitwise NOT)"

    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        return cv2.bitwise_not(image)


# Registry of all built-in steps for easy import
BUILTIN_STEPS = [
    GrayscaleStep,
    GaussianBlurStep,
    AdaptiveThresholdStep,
    BinaryThresholdStep,
    SobelGradientStep,
    CannyEdgeStep,
    MorphologyStep,
    MedianBlurStep,
    BilateralFilterStep,
    ContrastStep,
    InvertStep,
]
