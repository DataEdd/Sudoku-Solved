"""
Export a trained PyTorch ``SudokuCNN`` checkpoint to the ONNX format used
by ``app/ml/recognizer.py::CNNRecognizer`` in production.

The committed ONNX file under ``app/ml/checkpoints/`` is stored in the
*external-data* ONNX format — a small protobuf header
(``sudoku_cnn.onnx``, ~24 KB) that references the weight tensors held
in a sibling file (``sudoku_cnn.onnx.data``, ~396 KB). This script
reproduces that exact layout so that, after a fresh ``python -m app.ml.train``,
a single ``python -m app.ml.export_onnx`` regenerates the ONNX artefacts
the FastAPI app ships with.

The ONNX Runtime session in ``recognizer.py`` expects:

    - input name:  ``image``
    - input shape: ``(N, 1, 28, 28)`` float32 with dynamic ``N``
    - output:      ``(N, 10)`` logits (softmax is applied on the Python side)

Usage:
    python -m app.ml.export_onnx                              # Uses defaults
    python -m app.ml.export_onnx --pth path/to/checkpoint.pth
    python -m app.ml.export_onnx --verify                     # Also runs
                                                                # numerical parity
                                                                # check vs PyTorch
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from app.ml.model import SudokuCNN

DEFAULT_PTH = Path("app/ml/checkpoints/sudoku_cnn.pth")
DEFAULT_ONNX = Path("app/ml/checkpoints/sudoku_cnn.onnx")
DEFAULT_DATA_NAME = "sudoku_cnn.onnx.data"
ONNX_OPSET = 17


def load_checkpoint(pth_path: Path) -> SudokuCNN:
    """Instantiate a SudokuCNN and load weights from a .pth checkpoint."""
    if not pth_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {pth_path}. "
            "Train one first with `python -m app.ml.train`."
        )
    model = SudokuCNN()
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def export_to_onnx(
    model: SudokuCNN,
    onnx_path: Path,
    data_filename: str = DEFAULT_DATA_NAME,
    opset: int = ONNX_OPSET,
) -> None:
    """Export the model to ONNX in the external-data format.

    This is a two-step process:
      1. ``torch.onnx.export`` writes a self-contained ONNX file with
         all weight tensors embedded inline.
      2. We load it via ``onnx`` and re-save it with the
         ``save_as_external_data=True`` option so the weight tensors
         move into a sidecar file (``sudoku_cnn.onnx.data``) and the
         header shrinks from ~420 KB back to ~24 KB.

    Step 2 is what makes the layout match the committed files and keeps
    the ``.onnx`` header small enough to diff cleanly in git.
    """
    dummy = torch.zeros(1, 1, 28, 28, dtype=torch.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: export as a self-contained protobuf with all weights inline.
    # We deliberately use the legacy TorchScript-based exporter
    # (``dynamo=False``) rather than the new ``torch.export.export`` path.
    # The dynamo path produces a protobuf where the large conv weights are
    # marked as external and stored in a scratch file that is deleted when
    # the export call returns, so the resulting ``.onnx`` is unloadable on
    # its own. The legacy path reliably embeds every tensor inline, which
    # is what we need before manually converting to the committed external-
    # data layout in step 2.
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    # Step 2: re-save with external data so weights live in a sidecar.
    # The onnx library needs THREE explicit calls to do this correctly:
    #   (a) convert_model_to_external_data — rewrite tensor proto pointers
    #       to reference the external file by name
    #   (b) write_external_data_tensors    — actually write the sidecar
    #       file to disk
    #   (c) onnx.save_model with save_as_external_data=False — write the
    #       header. We pass False here because the tensors are already
    #       externalised by (a) and written by (b); if we passed True,
    #       onnx.save_model would try to externalise them again and silently
    #       produce an inconsistent state.
    import onnx
    from onnx.external_data_helper import (
        convert_model_to_external_data,
        write_external_data_tensors,
    )

    # Clear any stale sidecar and header from previous runs
    data_path = onnx_path.parent / data_filename
    if data_path.exists():
        data_path.unlink()

    model_proto = onnx.load(str(onnx_path), load_external_data=False)
    convert_model_to_external_data(
        model_proto,
        all_tensors_to_one_file=True,
        location=data_filename,
        size_threshold=1024,  # tensors < 1 KB stay inline
        convert_attribute=False,
    )
    write_external_data_tensors(model_proto, str(onnx_path.parent))
    onnx.save_model(model_proto, str(onnx_path), save_as_external_data=False)


def verify_parity(
    pth_model: SudokuCNN,
    onnx_path: Path,
    n_samples: int = 16,
    tolerance: float = 1e-4,
) -> None:
    """Run the same random inputs through PyTorch and ONNX Runtime;
    fail if the max absolute difference exceeds ``tolerance``.

    1e-4 is the standard FP32 parity tolerance for ONNX Runtime conversions —
    the two runtimes use slightly different kernel implementations and the
    residual noise floor is ~1e-5 relative to logit magnitudes of order 1,
    which is well below anything that affects argmax / top-k classification.
    """
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    batch = rng.standard_normal((n_samples, 1, 28, 28)).astype(np.float32)

    with torch.no_grad():
        pt_logits = pth_model(torch.from_numpy(batch)).numpy()

    session = ort.InferenceSession(str(onnx_path))
    (onnx_logits,) = session.run(None, {"image": batch})

    max_abs_diff = float(np.max(np.abs(pt_logits - onnx_logits)))
    mean_abs_diff = float(np.mean(np.abs(pt_logits - onnx_logits)))

    print(f"  parity: max_abs_diff = {max_abs_diff:.2e}, mean_abs_diff = {mean_abs_diff:.2e}")
    if max_abs_diff > tolerance:
        raise RuntimeError(
            f"PyTorch and ONNX outputs disagree by {max_abs_diff:.2e} "
            f"(tolerance {tolerance:.2e})"
        )
    print("  PyTorch ↔ ONNX parity OK")


def report_sizes(onnx_path: Path, data_filename: str) -> None:
    """Print the header + sidecar sizes and the total footprint."""
    header_bytes = onnx_path.stat().st_size
    data_path = onnx_path.parent / data_filename
    data_bytes = data_path.stat().st_size if data_path.exists() else 0
    total = header_bytes + data_bytes
    print()
    print("  Exported ONNX footprint:")
    print(f"    {onnx_path.name:<26} {header_bytes:>10,} B  ({header_bytes / 1024:6.1f} KB)")
    if data_bytes:
        print(f"    {data_filename:<26} {data_bytes:>10,} B  ({data_bytes / 1024:6.1f} KB)")
    print(f"    {'total':<26} {total:>10,} B  ({total / 1024:6.1f} KB)")
    if data_bytes == 0:
        print(
            "  NOTE: no external-data sidecar was produced — all weights "
            "are inline in the .onnx file."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a trained SudokuCNN .pth checkpoint to the "
                    "external-data ONNX format used by CNNRecognizer."
    )
    parser.add_argument(
        "--pth", type=Path, default=DEFAULT_PTH,
        help=f"Path to the input PyTorch checkpoint (default: {DEFAULT_PTH})",
    )
    parser.add_argument(
        "--onnx", type=Path, default=DEFAULT_ONNX,
        help=f"Path to the output ONNX file (default: {DEFAULT_ONNX})",
    )
    parser.add_argument(
        "--data-filename", type=str, default=DEFAULT_DATA_NAME,
        help=(
            f"Sidecar file name written next to the .onnx "
            f"(default: {DEFAULT_DATA_NAME})"
        ),
    )
    parser.add_argument(
        "--opset", type=int, default=ONNX_OPSET,
        help=f"ONNX opset version (default: {ONNX_OPSET})",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run a numerical parity check between PyTorch and ONNX Runtime "
             "after the export.",
    )
    args = parser.parse_args()

    print(f"Loading PyTorch checkpoint: {args.pth}")
    model = load_checkpoint(args.pth)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters")

    print(f"Exporting to ONNX (opset {args.opset}): {args.onnx}")
    export_to_onnx(
        model,
        args.onnx,
        data_filename=args.data_filename,
        opset=args.opset,
    )

    report_sizes(args.onnx, args.data_filename)

    if args.verify:
        print()
        print("Verifying PyTorch ↔ ONNX Runtime numerical parity...")
        verify_parity(model, args.onnx)


if __name__ == "__main__":
    main()
