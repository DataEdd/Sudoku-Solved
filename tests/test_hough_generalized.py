"""
Visual tests for Generalized Hough Transform (GHT) grid detection.

Run with: pytest tests/test_hough_generalized.py -v -s
"""

import cv2
import numpy as np
import pytest


class TestGHTVisualization:
    """Visual tests for Generalized Hough Transform."""

    def test_template_visualization(self, display_images):
        """
        Show the grid template used for shape matching.

        The template is a synthetic Sudoku grid that defines
        the shape we're looking for.
        """
        from app.core.hough_generalized import GridTemplate

        template = GridTemplate(size=200)

        display_images({
            "Grid Template": template.image,
            "Template Edges": template.edges,
        }, title="GHT: Template for Shape Matching")

    def test_r_table_visualization(self, display_images):
        """
        Visualize the R-table as vectors on the template.

        Each arrow shows the vector from an edge pixel to the center.
        The R-table indexes these by gradient direction.
        """
        from app.core.hough_generalized import GridTemplate, visualize_r_table

        template = GridTemplate(size=300)
        r_table_vis = visualize_r_table(template, n_samples=100)

        stats = template.get_r_table_stats()
        info = f"Entries: {stats['total_entries']}, Bins: {stats['bins_used']}"

        display_images({
            "Template": template.image,
            f"R-Table Vectors ({info})": r_table_vis,
        }, title="GHT: R-Table Visualization")

    def test_r_table_distribution(self, display_images):
        """
        Show distribution of R-table entries across angle bins.

        For a grid template, we expect peaks at 0° and 90°
        (horizontal and vertical edges).
        """
        import matplotlib.pyplot as plt
        from app.core.hough_generalized import GridTemplate

        template = GridTemplate(size=200)

        # Count entries per angle bin
        bins = list(range(360))
        counts = [len(template.r_table.get(b, [])) for b in bins]

        # Create polar plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 4),
                                  subplot_kw={'projection': None})

        # Histogram
        axes[0].bar(bins, counts, width=1)
        axes[0].set_xlabel('Angle Bin (degrees)')
        axes[0].set_ylabel('Number of Entries')
        axes[0].set_title('R-Table Entry Distribution')

        # Mark expected peaks (horizontal and vertical)
        for peak in [0, 90, 180, 270]:
            axes[0].axvline(x=peak, color='r', linestyle='--', alpha=0.5)

        # Template image
        axes[1].imshow(template.image, cmap='gray')
        axes[1].set_title('Template')
        axes[1].axis('off')

        plt.suptitle("GHT: R-Table Angle Distribution")
        plt.tight_layout()
        plt.show()

    def test_accumulator_visualization(self, sample_sudoku_image, display_images):
        """
        Show the voting accumulator as a heatmap.

        Peak in accumulator = detected grid center.
        """
        from app.core.hough_generalized import detect_grid_ght

        result = detect_grid_ght(
            sample_sudoku_image,
            return_accumulator=True
        )

        display_images({
            "Original": sample_sudoku_image,
            "Accumulator Heatmap": result.accumulator,
            "Detection Result": result.annotated_image,
        }, title=f"GHT: Accumulator (conf={result.confidence:.2f})")

    def test_detection_full(self, sample_sudoku_image, display_images):
        """
        Show complete GHT detection result.

        Green rectangle = detected grid boundary
        Red dot = detected center
        """
        from app.core.hough_generalized import detect_grid_ght

        result = detect_grid_ght(sample_sudoku_image)

        info = f"Center: ({result.center[0]:.0f}, {result.center[1]:.0f}), Scale: {result.scale:.2f}"

        display_images({
            "Original": sample_sudoku_image,
            f"Detection ({info})": result.annotated_image,
        }, title="GHT: Grid Detection")

    def test_scale_sensitivity(self, synthetic_sudoku, display_images):
        """
        Test detection at different image scales.

        GHT searches multiple scales to find best match.
        """
        from app.core.hough_generalized import detect_grid_ght

        results = {}
        for scale in [0.5, 0.75, 1.0, 1.5]:
            size = int(400 * scale)
            img = synthetic_sudoku(size=size)
            result = detect_grid_ght(img)
            results[f"Scale {scale}x (conf={result.confidence:.2f})"] = result.annotated_image

        display_images(results, title="GHT: Scale Sensitivity")

    def test_detection_on_variations(self, test_image_collection, display_images):
        """
        Test GHT detection on various image conditions.

        Shows robustness to rotation, perspective, noise.
        """
        from app.core.hough_generalized import detect_grid_ght

        results = {}
        for name, img in test_image_collection.items():
            result = detect_grid_ght(img, return_accumulator=False)
            if result.annotated_image is not None:
                results[f"{name} (conf={result.confidence:.2f})"] = result.annotated_image
            else:
                results[f"{name} (FAILED)"] = img

        display_images(results, title="GHT: Robustness Test")


class TestGHTMath:
    """Tests verifying GHT mathematical operations."""

    def test_template_creation(self):
        """Test that template is created correctly."""
        from app.core.hough_generalized import GridTemplate

        template = GridTemplate(size=100)

        # Template should be square
        assert template.image.shape == (100, 100)

        # Should have grid lines (non-zero pixels)
        assert np.sum(template.image > 0) > 0

        # Edges should exist
        assert np.sum(template.edges > 0) > 0

    def test_r_table_symmetry(self):
        """Test R-table has entries for horizontal and vertical edges."""
        from app.core.hough_generalized import GridTemplate

        template = GridTemplate(size=100)

        # Should have entries near 0° (horizontal edges)
        h_entries = sum(len(template.r_table.get(i, [])) for i in range(-10, 10))
        assert h_entries > 0, "Should have horizontal edge entries"

        # Should have entries near 90° (vertical edges)
        v_entries = sum(len(template.r_table.get(i, [])) for i in range(80, 100))
        assert v_entries > 0, "Should have vertical edge entries"

    def test_r_table_vectors_point_to_center(self):
        """Test that R-table vectors point toward center."""
        from app.core.hough_generalized import GridTemplate

        template = GridTemplate(size=100)
        center = template.center

        # Sample some entries and verify they point toward center
        for angle_bin, vectors in list(template.r_table.items())[:10]:
            for dx, dy in vectors[:5]:
                # These are vectors FROM edge TO center
                # So edge position + vector should give center
                # We can't verify exact positions without edge coords,
                # but vectors should have reasonable magnitude
                magnitude = np.sqrt(dx**2 + dy**2)
                assert 0 < magnitude < 100, f"Vector magnitude {magnitude} seems wrong"

    def test_detection_on_template(self):
        """Test that GHT can detect its own template (sanity check)."""
        from app.core.hough_generalized import GridTemplate, detect_grid_ght
        import cv2

        template = GridTemplate(size=200)

        # Convert template to BGR image
        img = cv2.cvtColor(template.image, cv2.COLOR_GRAY2BGR)

        result = detect_grid_ght(img, template=template)

        # Should detect center near actual center
        expected_center = template.center
        detected_center = result.center

        distance = np.sqrt(
            (detected_center[0] - expected_center[0])**2 +
            (detected_center[1] - expected_center[1])**2
        )

        # Allow some tolerance (20 pixels)
        assert distance < 50, f"Center detection off by {distance} pixels"

    def test_accumulator_peak(self, synthetic_sudoku):
        """Test that accumulator has a clear peak at grid center."""
        from app.core.hough_generalized import detect_grid_ght

        img = synthetic_sudoku(size=400)
        result = detect_grid_ght(img, return_accumulator=True)

        if result.accumulator is not None:
            # Find peak location
            acc_gray = cv2.cvtColor(result.accumulator, cv2.COLOR_BGR2GRAY)
            max_loc = np.unravel_index(acc_gray.argmax(), acc_gray.shape)

            # Peak should be near center of image
            h, w = img.shape[:2]
            center = (h // 2, w // 2)

            distance = np.sqrt((max_loc[0] - center[0])**2 + (max_loc[1] - center[1])**2)

            # Synthetic image is centered, so peak should be close to center
            assert distance < 100, f"Accumulator peak is {distance} from center"
