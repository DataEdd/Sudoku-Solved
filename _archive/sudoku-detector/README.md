# Sudoku Grid Detector

A Python library for detecting and extracting sudoku grids from images.

## Features

- Detects sudoku grids at arbitrary angles (up to 45° from horizontal)
- Handles variable lighting conditions using CLAHE normalization
- Returns the most centered sudoku if multiple grids are present
- Outputs perspective-corrected 450x450 images

## Detection Approach

The detector uses a dual-path approach for robust detection:

1. **Path A - Contour Detection**: Finds quadrilateral contours that match sudoku grid characteristics
2. **Path B - Hough Line Detection**: Detects grid lines and computes corners from line intersections

The best result is selected based on a centeredness score.

## Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from src.config import DetectionConfig

# Create configuration with default values
config = DetectionConfig()

# Or customize parameters
config = DetectionConfig(
    target_size=800,
    clahe_clip_limit=3.0,
    output_size=450
)
```

## Project Structure

```
sudoku-detector/
├── src/
│   ├── __init__.py
│   └── config.py
├── tests/
├── test_images/
├── requirements.txt
├── .gitignore
└── README.md
```

## Dependencies

- opencv-python-headless >= 4.5.0, < 5.0.0
- numpy >= 1.19.0, < 2.0.0
- pytest (for testing)

## Testing

```bash
pytest tests/
```

## License

MIT
