"""Sudoku extraction and solving API endpoints."""

import base64
from typing import Any, Dict, List

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from app.core.extraction import (
    detect_grid,
    extract_cells,
    find_grid_contour,
    order_points,
    perspective_transform,
    recognize_cells,
)
from app.core.solver import solve
from app.core.verifier import validate_puzzle, verify_solution
from app.models.schemas import ExtractResponse, SolveRequest, SolveResponse


def _encode_image(img: np.ndarray, fmt: str = ".jpg") -> str:
    """Encode a cv2 image to base64 data URI."""
    _, buf = cv2.imencode(fmt, img)
    b64 = base64.b64encode(buf).decode("utf-8")
    mime = "image/jpeg" if fmt == ".jpg" else "image/png"
    return f"data:{mime};base64,{b64}"

router = APIRouter()


@router.post("/extract", response_model=ExtractResponse)
async def extract_grid(file: UploadFile = File(...)):
    """Extract Sudoku grid from uploaded image using CNN OCR."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return ExtractResponse(
                success=False,
                message="Invalid image file",
            )

        # Detection (4-step fallback chain)
        corners, confidence = detect_grid(image)

        if corners is None:
            return ExtractResponse(
                success=False,
                message="Could not detect Sudoku grid in image.",
            )

        # Cell extraction + OCR
        warped = perspective_transform(image, corners.reshape(4, 1, 2).astype(np.int32))
        cells = extract_cells(warped)
        grid, confidence_map = recognize_cells(cells)

        return ExtractResponse(
            success=True,
            grid=grid,
            confidence_map=confidence_map,
            message="Grid extracted successfully",
        )

    except Exception as e:
        return ExtractResponse(
            success=False,
            message=f"Error processing image: {str(e)}",
        )


@router.post("/solve", response_model=SolveResponse)
async def solve_sudoku(data: SolveRequest):
    """Solve a Sudoku puzzle via MRV-ordered backtracking."""
    grid = data.grid

    # Validate puzzle before solving
    valid, errors = validate_puzzle(grid)
    if not valid:
        return SolveResponse(
            success=False,
            message=f"Invalid puzzle: {'; '.join(errors)}",
        )

    solution, nodes_explored, success, elapsed_ms = solve(grid)

    if success and verify_solution(solution):
        return SolveResponse(
            success=True,
            solution=solution,
            nodes_explored=nodes_explored,
            solve_time_ms=round(elapsed_ms, 1),
            message=f"Solved in {nodes_explored} nodes ({elapsed_ms:.1f}ms)",
        )
    else:
        return SolveResponse(
            success=False,
            solution=solution,
            nodes_explored=nodes_explored,
            solve_time_ms=round(elapsed_ms, 1),
            message="Could not find valid solution.",
        )


@router.post("/debug")
async def debug_pipeline(
    file: UploadFile = File(...),
    blur_k: int = Form(5),
    block_size: int = Form(11),
    thresh_c: int = Form(2),
    epsilon: float = Form(0.02),
    corners: str = Form(""),
    cell_margin: int = Form(10),
    empty_thresh: float = Form(0.03),
    conf_thresh: float = Form(0.7),
):
    """Debug endpoint: return every intermediate pipeline step with tunable params."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse({"error": "Invalid image file"}, status_code=400)

    # Sanitize odd-only params
    blur_k = max(1, blur_k) | 1
    block_size = max(3, block_size) | 1

    result: Dict[str, Any] = {}
    h, w = image.shape[:2]
    result["image_w"] = w
    result["image_h"] = h

    # ── Step 1: Original ──
    result["original_b64"] = _encode_image(image)

    # ── Step 2: Preprocessing sub-steps ──
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result["grayscale_b64"] = _encode_image(gray)

    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    result["blurred_b64"] = _encode_image(blurred)

    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, thresh_c
    )
    result["binary_b64"] = _encode_image(binary)

    # ── Step 3: Contour detection ──
    contours_all, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contours_all, key=cv2.contourArea, reverse=True)[:15]

    # Draw all contours with area labels
    all_contours_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for i, cnt in enumerate(contours_sorted):
        color = (255, 100, 50) if i > 0 else (0, 255, 0)
        cv2.drawContours(all_contours_img, [cnt], -1, color, 2)
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            label = f"#{i+1} {area:.0f}"
            cv2.putText(all_contours_img, label, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    result["all_contours_b64"] = _encode_image(all_contours_img)

    # Find winning quad or use manual corners
    import json as _json

    manual_corners = None
    if corners:
        try:
            manual_corners = np.array(_json.loads(corners), dtype=np.float32).reshape(4, 2)
        except Exception:
            pass

    if manual_corners is not None:
        quad = manual_corners
        result["corner_source"] = "manual"
    else:
        # Auto-detect using the same scoring as production
        best_contour = find_grid_contour(binary, epsilon=epsilon)
        if best_contour is not None:
            quad = best_contour.reshape(4, 2).astype(np.float32)
        else:
            quad = None
        result["corner_source"] = "auto"

    if quad is None:
        result["error"] = "No quadrilateral found. Try adjusting preprocessing params or epsilon."
        return JSONResponse(result)

    # Order corners
    ordered = order_points(quad.reshape(4, 1, 2))
    result["corners"] = ordered.tolist()

    # Draw selected contour + corners on original
    overlay = image.copy()
    pts_int = ordered.astype(np.int32)
    cv2.polylines(overlay, [pts_int], True, (0, 255, 0), 3)
    corner_colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (255, 0, 0)]
    corner_labels = ["TL", "TR", "BR", "BL"]
    for pt, color, label in zip(ordered, corner_colors, corner_labels):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(overlay, (x, y), 8, color, -1)
        cv2.putText(overlay, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    result["selected_contour_b64"] = _encode_image(overlay)

    # ── Step 4: Perspective warp ──
    warped = perspective_transform(image, quad.reshape(4, 1, 2))
    result["warped_b64"] = _encode_image(warped)

    # ── Step 5: Cell OCR ──
    cells = extract_cells(warped)

    from app.ml.recognizer import CNNRecognizer

    rec = CNNRecognizer(
        confidence_threshold=conf_thresh,
        empty_threshold=empty_thresh,
    )

    # Override margin for debugging
    original_preprocess = rec._preprocess

    def _custom_preprocess(cell_img: np.ndarray) -> np.ndarray:
        if len(cell_img.shape) == 3:
            cell_img = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        ch, cw = cell_img.shape
        my = max(1, ch * cell_margin // 100)
        mx = max(1, cw * cell_margin // 100)
        cell_img = cell_img[my:-my, mx:-mx]
        cell_img = cv2.resize(cell_img, (28, 28), interpolation=cv2.INTER_AREA)
        _, cell_img = cv2.threshold(cell_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return cell_img

    rec._preprocess = _custom_preprocess
    predictions = rec.predict_batch(cells)

    cell_details: List[Dict[str, Any]] = []
    for idx, (cell_img, (digit, conf)) in enumerate(zip(cells, predictions)):
        raw_b64 = _encode_image(cell_img, ".png")
        processed = _custom_preprocess(cell_img)
        proc_b64 = _encode_image(processed, ".png")
        white_ratio = float(np.sum(processed == 255) / processed.size)
        is_empty = white_ratio < empty_thresh

        cell_details.append({
            "row": idx // 9,
            "col": idx % 9,
            "raw_b64": raw_b64,
            "processed_b64": proc_b64,
            "digit": digit,
            "confidence": round(conf, 4),
            "white_ratio": round(white_ratio, 4),
            "is_empty": bool(is_empty),
        })

    result["cells"] = cell_details

    # ── Step 6: Grid + Solve ──
    grid: List[List[int]] = []
    confidence_map: List[List[float]] = []
    for i in range(9):
        grid.append([cell_details[i * 9 + j]["digit"] for j in range(9)])
        confidence_map.append([cell_details[i * 9 + j]["confidence"] for j in range(9)])
    result["grid"] = grid
    result["confidence_map"] = confidence_map

    try:
        solution, nodes_explored, success, elapsed_ms = solve(grid)
        if success and verify_solution(solution):
            result["solution"] = solution
            result["solve_success"] = True
            result["solve_time_ms"] = round(elapsed_ms, 1)
            result["solve_nodes"] = nodes_explored
        else:
            result["solution"] = None
            result["solve_success"] = False
    except Exception:
        result["solution"] = None
        result["solve_success"] = False

    return JSONResponse(result)
