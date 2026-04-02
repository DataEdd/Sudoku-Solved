"""
Comparison tests between Standard Hough and Generalized Hough Transform.

Run with: pytest tests/test_pipeline_comparison.py -v -s
"""

import time

import cv2
import numpy as np
import pytest


class TestPipelineComparison:
    """Compare Standard Hough vs Generalized Hough Transform."""

    def test_side_by_side_comparison(self, sample_sudoku_image, display_images):
        """
        Side-by-side comparison of both detection methods.

        Shows how each method handles the same input image.
        """
        from app.core.hough_standard import detect_grid_standard
        from app.core.hough_generalized import detect_grid_ght

        result_standard = detect_grid_standard(sample_sudoku_image, return_visualization=True)
        result_ght = detect_grid_ght(sample_sudoku_image, return_accumulator=True)

        display_images({
            "Original": sample_sudoku_image,
            f"Standard (conf={result_standard.confidence:.2f})": result_standard.annotated_image,
            f"GHT (conf={result_ght.confidence:.2f})": result_ght.annotated_image,
        }, title="Method Comparison: Standard vs Generalized Hough")

    def test_robustness_comparison(self, test_image_collection, display_images):
        """
        Compare robustness on challenging images.

        Tests: clean, rotated, perspective, noisy images.
        """
        from app.core.hough_standard import detect_grid_standard
        from app.core.hough_generalized import detect_grid_ght

        import matplotlib.pyplot as plt

        n_images = len(test_image_collection)
        fig, axes = plt.subplots(3, n_images, figsize=(4 * n_images, 10))

        for i, (name, img) in enumerate(test_image_collection.items()):
            result_std = detect_grid_standard(img)
            result_ght = detect_grid_ght(img)

            # Original
            axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0, i].set_title(name)
            axes[0, i].axis('off')

            # Standard Hough
            if result_std.annotated_image is not None:
                axes[1, i].imshow(cv2.cvtColor(result_std.annotated_image, cv2.COLOR_BGR2RGB))
            else:
                axes[1, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[1, i].set_title(f"Std: {result_std.confidence:.2f}")
            axes[1, i].axis('off')

            # GHT
            if result_ght.annotated_image is not None:
                axes[2, i].imshow(cv2.cvtColor(result_ght.annotated_image, cv2.COLOR_BGR2RGB))
            else:
                axes[2, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[2, i].set_title(f"GHT: {result_ght.confidence:.2f}")
            axes[2, i].axis('off')

        axes[0, 0].set_ylabel('Original', fontsize=12)
        axes[1, 0].set_ylabel('Standard Hough', fontsize=12)
        axes[2, 0].set_ylabel('Generalized Hough', fontsize=12)

        plt.suptitle("Robustness Comparison", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def test_speed_comparison(self, synthetic_sudoku):
        """
        Compare processing speed of both methods.

        Standard Hough is typically faster but less robust.
        """
        from app.core.hough_standard import detect_grid_standard
        from app.core.hough_generalized import detect_grid_ght

        import matplotlib.pyplot as plt

        sizes = [200, 300, 400, 500, 600]
        std_times = []
        ght_times = []

        n_runs = 3

        for size in sizes:
            img = synthetic_sudoku(size=size)

            # Standard Hough timing
            times = []
            for _ in range(n_runs):
                start = time.time()
                detect_grid_standard(img, return_visualization=False)
                times.append(time.time() - start)
            std_times.append(np.mean(times))

            # GHT timing
            times = []
            for _ in range(n_runs):
                start = time.time()
                detect_grid_ght(img, return_accumulator=False)
                times.append(time.time() - start)
            ght_times.append(np.mean(times))

        # Plot results
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sizes, std_times, 'b-o', label='Standard Hough')
        ax.plot(sizes, ght_times, 'r-s', label='Generalized Hough')
        ax.set_xlabel('Image Size (pixels)')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Speed Comparison')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()

        print("\nSpeed Comparison Results:")
        print("-" * 50)
        for size, std_t, ght_t in zip(sizes, std_times, ght_times):
            ratio = ght_t / std_t if std_t > 0 else float('inf')
            print(f"Size {size}x{size}: Std={std_t*1000:.1f}ms, GHT={ght_t*1000:.1f}ms, Ratio={ratio:.1f}x")

    def test_rotation_sensitivity(self, synthetic_sudoku, display_images):
        """
        Compare how each method handles rotation.

        GHT should be more robust to rotation.
        """
        from app.core.hough_standard import detect_grid_standard
        from app.core.hough_generalized import detect_grid_ght

        import matplotlib.pyplot as plt

        rotations = [0, 5, 10, 15, 20, 30, 45]

        fig, axes = plt.subplots(3, len(rotations), figsize=(3 * len(rotations), 8))

        std_confs = []
        ght_confs = []

        for i, rotation in enumerate(rotations):
            img = synthetic_sudoku(size=400, rotation=rotation)

            result_std = detect_grid_standard(img)
            result_ght = detect_grid_ght(img)

            std_confs.append(result_std.confidence)
            ght_confs.append(result_ght.confidence)

            # Original
            axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0, i].set_title(f'{rotation}°')
            axes[0, i].axis('off')

            # Standard
            if result_std.annotated_image is not None:
                axes[1, i].imshow(cv2.cvtColor(result_std.annotated_image, cv2.COLOR_BGR2RGB))
            else:
                axes[1, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[1, i].set_title(f'{result_std.confidence:.2f}')
            axes[1, i].axis('off')

            # GHT
            if result_ght.annotated_image is not None:
                axes[2, i].imshow(cv2.cvtColor(result_ght.annotated_image, cv2.COLOR_BGR2RGB))
            else:
                axes[2, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[2, i].set_title(f'{result_ght.confidence:.2f}')
            axes[2, i].axis('off')

        axes[0, 0].set_ylabel('Original', fontsize=12)
        axes[1, 0].set_ylabel('Standard', fontsize=12)
        axes[2, 0].set_ylabel('GHT', fontsize=12)

        plt.suptitle("Rotation Sensitivity", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Print summary
        print("\nRotation Sensitivity Summary:")
        print("-" * 50)
        print("Rotation | Standard | GHT")
        print("-" * 50)
        for rot, std_c, ght_c in zip(rotations, std_confs, ght_confs):
            winner = "GHT" if ght_c > std_c else "Std" if std_c > ght_c else "Tie"
            print(f"{rot:6}°  | {std_c:.3f}    | {ght_c:.3f}  ({winner})")

    def test_noise_sensitivity(self, synthetic_sudoku):
        """
        Compare how each method handles noise.
        """
        from app.core.hough_standard import detect_grid_standard
        from app.core.hough_generalized import detect_grid_ght

        import matplotlib.pyplot as plt

        noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

        fig, axes = plt.subplots(3, len(noise_levels), figsize=(3 * len(noise_levels), 8))

        for i, noise in enumerate(noise_levels):
            img = synthetic_sudoku(size=400, noise_level=noise)

            result_std = detect_grid_standard(img)
            result_ght = detect_grid_ght(img)

            # Original
            axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0, i].set_title(f'Noise={noise}')
            axes[0, i].axis('off')

            # Standard
            if result_std.annotated_image is not None:
                axes[1, i].imshow(cv2.cvtColor(result_std.annotated_image, cv2.COLOR_BGR2RGB))
            else:
                axes[1, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[1, i].set_title(f'{result_std.confidence:.2f}')
            axes[1, i].axis('off')

            # GHT
            if result_ght.annotated_image is not None:
                axes[2, i].imshow(cv2.cvtColor(result_ght.annotated_image, cv2.COLOR_BGR2RGB))
            else:
                axes[2, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[2, i].set_title(f'{result_ght.confidence:.2f}')
            axes[2, i].axis('off')

        axes[0, 0].set_ylabel('Original', fontsize=12)
        axes[1, 0].set_ylabel('Standard', fontsize=12)
        axes[2, 0].set_ylabel('GHT', fontsize=12)

        plt.suptitle("Noise Sensitivity", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_full_extraction_pipeline(self, synthetic_sudoku, display_images):
        """
        Test complete pipeline from image to extracted grid.
        """
        from app.core.hough_standard import detect_grid_standard
        from app.core.preprocessing import preprocess_full

        img = synthetic_sudoku(size=500, fill_digits=True)

        # Preprocessing
        preprocess_result = preprocess_full(img)

        # Grid detection
        detection_result = detect_grid_standard(img)

        display_images({
            "1. Original": preprocess_result.original,
            "2. Threshold": preprocess_result.threshold,
            "3. Detected Grid": detection_result.annotated_image,
        }, title="Full Extraction Pipeline")

        # Verify we got a grid
        assert len(detection_result.h_positions) == 10, "Should have 10 horizontal lines"
        assert len(detection_result.v_positions) == 10, "Should have 10 vertical lines"
        assert detection_result.intersections.shape == (10, 10, 2), "Should have 10x10 intersections"

    def test_pipeline_with_method_selection(self, sample_sudoku_image):
        """
        Test that we can select detection method.
        """
        from app.core.hough_standard import detect_grid_standard
        from app.core.hough_generalized import detect_grid_ght

        # Both should return valid results
        result_std = detect_grid_standard(sample_sudoku_image)
        result_ght = detect_grid_ght(sample_sudoku_image)

        # Both should have a confidence score
        assert 0 <= result_std.confidence <= 1
        assert 0 <= result_ght.confidence <= 1

        print(f"\nDetection Comparison:")
        print(f"  Standard Hough: confidence = {result_std.confidence:.3f}")
        print(f"  Generalized HT: confidence = {result_ght.confidence:.3f}")
