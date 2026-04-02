"""
CNN-based digit recognizer implementing the DigitRecognizer protocol.

Supports two backends:
- ONNX Runtime (default, lightweight, used in deployment)
- PyTorch (used for training/development)

Falls back automatically: ONNX -> PyTorch -> error.
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

DEFAULT_ONNX = Path("app/ml/checkpoints/sudoku_cnn.onnx")
DEFAULT_PTH = Path("app/ml/checkpoints/sudoku_cnn.pth")


class CNNRecognizer:
    """Digit recognizer using a trained SudokuCNN model.

    Implements the DigitRecognizer protocol from app.core.ocr.
    Prefers ONNX Runtime (50MB dep) over PyTorch (2GB dep) for deployment.
    """

    def __init__(
        self,
        onnx_path: str | Path = DEFAULT_ONNX,
        pth_path: str | Path = DEFAULT_PTH,
        confidence_threshold: float = 0.7,
        empty_threshold: float = 0.03,
        device: str = "auto",
    ):
        self.confidence_threshold = confidence_threshold
        self.empty_threshold = empty_threshold
        self.backend = None
        self._onnx_session = None
        self._torch_model = None
        self._torch_device = None

        # Try ONNX first (lightweight), fall back to PyTorch
        if Path(onnx_path).exists():
            try:
                import onnxruntime as ort

                self._onnx_session = ort.InferenceSession(str(onnx_path))
                self.backend = "onnx"
            except ImportError:
                pass

        if self.backend is None and Path(pth_path).exists():
            try:
                import torch

                from app.ml.model import SudokuCNN

                if device == "auto":
                    if torch.backends.mps.is_available():
                        self._torch_device = torch.device("mps")
                    elif torch.cuda.is_available():
                        self._torch_device = torch.device("cuda")
                    else:
                        self._torch_device = torch.device("cpu")
                else:
                    self._torch_device = torch.device(device)

                self._torch_model = SudokuCNN()
                checkpoint = torch.load(
                    pth_path, map_location=self._torch_device, weights_only=True
                )
                self._torch_model.load_state_dict(checkpoint["model_state_dict"])
                self._torch_model.to(self._torch_device)
                self._torch_model.eval()
                self.backend = "pytorch"
            except ImportError:
                pass

        if self.backend is None:
            raise RuntimeError(
                "No CNN model available. Need either "
                f"{onnx_path} + onnxruntime, or {pth_path} + torch."
            )

    @property
    def device(self) -> str:
        if self.backend == "onnx":
            return "cpu (onnx)"
        return str(self._torch_device)

    def _preprocess(self, cell: np.ndarray) -> np.ndarray:
        """Preprocess a cell image to 28x28 grayscale, white-on-black."""
        if len(cell.shape) == 3:
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

        h, w = cell.shape
        margin_y, margin_x = max(1, h // 10), max(1, w // 10)
        cell = cell[margin_y:-margin_y, margin_x:-margin_x]

        cell = cv2.resize(cell, (28, 28), interpolation=cv2.INTER_AREA)
        _, cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return cell

    def _is_empty(self, processed: np.ndarray) -> bool:
        white_ratio = np.sum(processed == 255) / processed.size
        return white_ratio < self.empty_threshold

    def _infer_onnx(self, batch: np.ndarray) -> np.ndarray:
        """Run ONNX inference. batch: (N, 1, 28, 28) float32 -> (N, 10) logits."""
        return self._onnx_session.run(None, {"image": batch})[0]

    def _infer_torch(self, batch: np.ndarray) -> np.ndarray:
        """Run PyTorch inference. batch: (N, 1, 28, 28) float32 -> (N, 10) logits."""
        import torch

        tensor = torch.from_numpy(batch).to(self._torch_device)
        with torch.no_grad():
            logits = self._torch_model(tensor)
        return logits.cpu().numpy()

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, cell: np.ndarray) -> Tuple[int, float]:
        processed = self._preprocess(cell)
        if self._is_empty(processed):
            return 0, 1.0

        batch = processed[np.newaxis, np.newaxis, :, :].astype(np.float32) / 255.0

        if self.backend == "onnx":
            logits = self._infer_onnx(batch)
        else:
            logits = self._infer_torch(batch)

        probs = self._softmax(logits)
        digit = int(probs[0].argmax())
        conf = float(probs[0].max())

        if conf < self.confidence_threshold:
            return 0, conf
        return digit, conf

    def predict_batch(self, cells: List[np.ndarray]) -> List[Tuple[int, float]]:
        results: List[Tuple[int, float]] = []
        batch_indices: List[int] = []
        batch_arrays: List[np.ndarray] = []

        for i, cell in enumerate(cells):
            processed = self._preprocess(cell)
            if self._is_empty(processed):
                results.append((0, 1.0))
            else:
                batch_arrays.append(processed.astype(np.float32) / 255.0)
                batch_indices.append(i)
                results.append(None)  # placeholder

        if batch_arrays:
            batch = np.stack(batch_arrays)[:, np.newaxis, :, :]  # (N, 1, 28, 28)

            if self.backend == "onnx":
                logits = self._infer_onnx(batch)
            else:
                logits = self._infer_torch(batch)

            probs = self._softmax(logits)
            predictions = probs.argmax(axis=1)
            confidences = probs.max(axis=1)

            for j, idx in enumerate(batch_indices):
                digit = int(predictions[j])
                conf = float(confidences[j])
                if conf < self.confidence_threshold:
                    results[idx] = (0, conf)
                else:
                    results[idx] = (digit, conf)

        return results
