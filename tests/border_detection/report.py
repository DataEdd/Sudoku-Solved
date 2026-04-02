"""
HTML report generator for border detection tests.

Generates self-contained HTML reports with embedded images.
"""

import base64
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .runner import TestRunResults, ImageTestResult


def encode_image_base64(image: np.ndarray, format: str = ".jpg") -> str:
    """
    Encode image as base64 data URI.

    Args:
        image: BGR or grayscale image
        format: Image format (".jpg", ".png")

    Returns:
        Data URI string for HTML img src
    """
    # Ensure image is in correct format
    if len(image.shape) == 2:
        # Grayscale - convert to BGR for encoding
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    _, buffer = cv2.imencode(format, image)
    b64 = base64.b64encode(buffer).decode("utf-8")

    mime = "image/jpeg" if format == ".jpg" else "image/png"
    return f"data:{mime};base64,{b64}"


def generate_html_report(
    results: TestRunResults,
    output_path: Path,
    include_debug_images: bool = True,
    max_debug_images: int = 6,
) -> None:
    """
    Generate HTML report from test results.

    Args:
        results: TestRunResults from test run
        output_path: Path to write HTML file
        include_debug_images: Include intermediate processing images
        max_debug_images: Maximum debug images to show per result
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html = _generate_report_html(results, include_debug_images, max_debug_images)

    with open(output_path, "w") as f:
        f.write(html)


def _generate_report_html(
    results: TestRunResults,
    include_debug_images: bool,
    max_debug_images: int,
) -> str:
    """Generate complete HTML document."""

    # Generate result cards
    result_cards = []
    for img_result in results.image_results:
        card = _generate_result_card(img_result, include_debug_images, max_debug_images)
        result_cards.append(card)

    results_html = "\n".join(result_cards)

    # Generate params table
    params_html = "\n".join(
        f'<div class="config-item"><span class="config-key">{k}:</span> '
        f'<span class="config-value">{v}</span></div>'
        for k, v in results.params.items()
    )

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Border Detection Report - {results.timestamp[:19]}</title>
    <style>
        :root {{
            --success: #22c55e;
            --failure: #ef4444;
            --warning: #f59e0b;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --border: #e2e8f0;
            --text: #1e293b;
            --text-muted: #64748b;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            margin: 0;
            padding: 20px;
            color: var(--text);
            line-height: 1.5;
        }}
        .header {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 8px 0;
            font-size: 1.5rem;
        }}
        .header .subtitle {{
            color: var(--text-muted);
            margin: 0 0 20px 0;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 16px;
        }}
        .stat {{
            background: var(--bg);
            padding: 16px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            line-height: 1.2;
        }}
        .stat-label {{
            color: var(--text-muted);
            font-size: 0.85em;
            margin-top: 4px;
        }}
        .stat.success .stat-value {{ color: var(--success); }}
        .stat.failure .stat-value {{ color: var(--failure); }}

        .config-section {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .config-section h2 {{
            margin: 0 0 16px 0;
            font-size: 1.1rem;
        }}
        .config-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .config-item {{
            background: var(--bg);
            padding: 8px 12px;
            border-radius: 6px;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.85em;
        }}
        .config-key {{ color: #6366f1; }}
        .config-value {{ color: #059669; }}

        .results-section h2 {{
            margin: 0 0 16px 0;
            font-size: 1.1rem;
        }}
        .results-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }}

        .result-card {{
            background: var(--card-bg);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .result-card.success {{ border-left: 4px solid var(--success); }}
        .result-card.failure {{ border-left: 4px solid var(--failure); }}

        .result-header {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .result-title {{
            font-weight: 600;
            font-size: 0.9em;
            margin: 0;
            max-width: 70%;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .result-badge {{
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.75em;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .badge-success {{ background: #dcfce7; color: #166534; }}
        .badge-failure {{ background: #fee2e2; color: #991b1b; }}

        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
            gap: 8px;
            padding: 12px;
        }}
        .image-item {{
            text-align: center;
        }}
        .image-item img {{
            width: 100%;
            border-radius: 4px;
            border: 1px solid var(--border);
            cursor: pointer;
            transition: transform 0.15s, box-shadow 0.15s;
        }}
        .image-item img:hover {{
            transform: scale(1.03);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .image-label {{
            font-size: 0.7em;
            color: var(--text-muted);
            margin-top: 4px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .result-meta {{
            padding: 12px 16px;
            background: var(--bg);
            font-size: 0.85em;
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
        }}
        .meta-item {{
            display: flex;
            justify-content: space-between;
        }}
        .meta-key {{ color: var(--text-muted); }}

        .lightbox {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.9);
            z-index: 1000;
            cursor: pointer;
            align-items: center;
            justify-content: center;
        }}
        .lightbox.active {{ display: flex; }}
        .lightbox img {{
            max-width: 90%;
            max-height: 90%;
            object-fit: contain;
            border-radius: 8px;
        }}
        .lightbox-close {{
            position: absolute;
            top: 20px;
            right: 20px;
            color: white;
            font-size: 2em;
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Border Detection Test Report</h1>
        <p class="subtitle">Method: <strong>{results.method}</strong> | {results.timestamp[:19]}</p>
        <div class="summary">
            <div class="stat success">
                <div class="stat-value">{results.success_count}</div>
                <div class="stat-label">Successful</div>
            </div>
            <div class="stat failure">
                <div class="stat-value">{results.failure_count}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat">
                <div class="stat-value">{results.success_rate * 100:.0f}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat">
                <div class="stat-value">{results.avg_time_ms:.0f}ms</div>
                <div class="stat-label">Avg Time</div>
            </div>
        </div>
    </div>

    <div class="config-section">
        <h2>Configuration</h2>
        <div class="config-grid">
            <div class="config-item">
                <span class="config-key">method:</span>
                <span class="config-value">{results.method}</span>
            </div>
            <div class="config-item">
                <span class="config-key">images:</span>
                <span class="config-value">{results.total}</span>
            </div>
            {params_html}
        </div>
    </div>

    <div class="results-section">
        <h2>Results ({results.total} images)</h2>
        <div class="results-grid">
            {results_html}
        </div>
    </div>

    <div class="lightbox" onclick="hideLightbox()">
        <span class="lightbox-close">&times;</span>
        <img src="" alt="Full size">
    </div>

    <script>
        function showLightbox(img) {{
            const lightbox = document.querySelector('.lightbox');
            lightbox.querySelector('img').src = img.src;
            lightbox.classList.add('active');
        }}
        function hideLightbox() {{
            document.querySelector('.lightbox').classList.remove('active');
        }}
        document.addEventListener('keydown', e => {{
            if (e.key === 'Escape') hideLightbox();
        }});
    </script>
</body>
</html>'''

    return html


def _generate_result_card(
    result: ImageTestResult,
    include_debug: bool,
    max_debug: int,
) -> str:
    """Generate HTML for a single result card."""

    status = "success" if result.detection_result.success else "failure"
    badge_class = "badge-success" if result.detection_result.success else "badge-failure"
    badge_text = "PASS" if result.detection_result.success else "FAIL"

    # Encode images
    original_b64 = encode_image_base64(result.original_image)
    detection_b64 = encode_image_base64(result.detection_image)

    # Build image grid
    image_items = [
        f'''<div class="image-item">
            <img src="{original_b64}" alt="Original" onclick="showLightbox(this)">
            <div class="image-label">Original</div>
        </div>''',
    ]

    # Add debug images
    if include_debug and result.detection_result.debug_images:
        debug_items = list(result.detection_result.debug_images.items())[:max_debug]
        for name, img in debug_items:
            img_b64 = encode_image_base64(img)
            # Clean up name for display
            display_name = name.replace("_", " ").title()
            if display_name[0].isdigit():
                display_name = display_name.split(" ", 1)[-1]
            image_items.append(
                f'''<div class="image-item">
                    <img src="{img_b64}" alt="{name}" onclick="showLightbox(this)">
                    <div class="image-label">{display_name}</div>
                </div>'''
            )

    # Add final detection
    image_items.append(
        f'''<div class="image-item">
            <img src="{detection_b64}" alt="Detection" onclick="showLightbox(this)">
            <div class="image-label">Detection</div>
        </div>'''
    )

    images_html = "\n".join(image_items)

    # Corners info
    corners_html = "N/A"
    if result.detection_result.corners is not None:
        c = result.detection_result.corners
        corners_html = f"TL({c[0][0]:.0f},{c[0][1]:.0f})"

    return f'''<div class="result-card {status}">
        <div class="result-header">
            <h3 class="result-title" title="{result.image_name}">{result.image_name}</h3>
            <span class="result-badge {badge_class}">{badge_text}</span>
        </div>
        <div class="image-grid">
            {images_html}
        </div>
        <div class="result-meta">
            <div class="meta-item">
                <span class="meta-key">Confidence:</span>
                <span>{result.detection_result.confidence:.2f}</span>
            </div>
            <div class="meta-item">
                <span class="meta-key">Time:</span>
                <span>{result.detection_result.execution_time_ms:.1f}ms</span>
            </div>
            <div class="meta-item">
                <span class="meta-key">Aug Level:</span>
                <span>{result.image_info.get('aug_level', 'N/A')}</span>
            </div>
            <div class="meta-item">
                <span class="meta-key">Corners:</span>
                <span>{corners_html}</span>
            </div>
        </div>
    </div>'''
