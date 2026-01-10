# Sudoku Solver

A web application that extracts Sudoku puzzles from images and solves them using simulated annealing.

## Features

- **Camera Capture** - Take a photo of a Sudoku puzzle directly from your device
- **Image Upload** - Upload an existing image of a puzzle
- **Automatic Grid Extraction** - Uses computer vision to detect and extract the puzzle grid
- **OCR Digit Recognition** - Reads digits from the extracted cells
- **Editable Grid** - Manually correct any misread digits before solving
- **Simulated Annealing Solver** - Solves puzzles using a probabilistic optimization algorithm

## Tech Stack

- **Backend:** Python, FastAPI, OpenCV, Pytesseract
- **Frontend:** HTML, CSS, JavaScript
- **Algorithm:** Simulated Annealing

## How It Works

### Image Processing Pipeline
1. Convert image to grayscale and apply adaptive thresholding
2. Detect the largest quadrilateral contour (the grid)
3. Apply perspective transform to get a top-down view
4. Split into 81 individual cells
5. Use Tesseract OCR to recognize digits

### Solver Algorithm
The simulated annealing solver:
- Minimizes an energy function (count of row + column conflicts)
- Generates neighbors by swapping non-fixed cells within 3x3 boxes
- Uses exponential cooling with periodic reheating to escape local minima

## Setup

### Prerequisites
- Python 3.8+
- Tesseract OCR

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr
```

### Installation

```bash
# Clone the repo
git clone https://github.com/DataEdd/Sudoku-Solved.git
cd Sudoku-Solved

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload
```

Visit `http://localhost:8000` in your browser.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve the frontend |
| `/api/extract` | POST | Extract grid from uploaded image |
| `/api/solve` | POST | Solve the provided grid |

## License

MIT
