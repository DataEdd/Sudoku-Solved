"""
Sudoku Solver - FastAPI server.

Extracts Sudoku puzzles from images using CNN OCR and solves them
with backtracking or simulated annealing.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from app.api.v1.endpoints.sudoku import router as sudoku_router

app = FastAPI(title="Sudoku Solver")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# API routes
app.include_router(sudoku_router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/debug", response_class=HTMLResponse)
async def debug_page(request: Request):
    """Serve the pipeline debug visualizer."""
    return templates.TemplateResponse(request=request, name="debug.html")


@app.get("/api/health")
async def health():
    """Health check with model status."""
    cnn_available = Path("app/ml/checkpoints/sudoku_cnn.pth").exists()
    return {
        "status": "ok",
        "ocr": "cnn" if cnn_available else "tesseract",
        "solvers": ["backtracking", "simulated_annealing"],
    }


@app.get("/sw.js")
async def service_worker():
    """Serve the service worker from root for proper scope."""
    return FileResponse("sw.js", media_type="application/javascript")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
