"""
CLI entry point for border detection testing.

Usage:
    python -m tests.border_detection --sample 5 --method simple_baseline
    python -m tests.border_detection --sample 10 --method all --output report.html
    python -m tests.border_detection --list-methods
"""

import argparse
import sys
import webbrowser
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Border Detection Testing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with 5 random images using simple_baseline method
  python -m tests.border_detection --sample 5 --method simple_baseline

  # Test with all methods for comparison
  python -m tests.border_detection --sample 10 --method all

  # Test specific images
  python -m tests.border_detection --images img1.jpg img2.jpg --method sobel_flood

  # List available methods
  python -m tests.border_detection --list-methods

  # Use custom parameters
  python -m tests.border_detection --sample 5 --method simple_baseline --params threshold=40

  # Filter by augmentation level
  python -m tests.border_detection --sample 5 --aug-level 0 100
        """
    )

    # Image selection (mutually exclusive)
    img_group = parser.add_mutually_exclusive_group()
    img_group.add_argument(
        "--sample", "-s",
        type=int,
        metavar="N",
        help="Randomly select N images from Examples/aug/"
    )
    img_group.add_argument(
        "--images", "-i",
        nargs="+",
        metavar="PATH",
        help="Specific image files to test"
    )
    img_group.add_argument(
        "--list-methods",
        action="store_true",
        help="List all available detection methods and exit"
    )

    # Detection method
    parser.add_argument(
        "--method", "-m",
        type=str,
        default="simple_baseline",
        help="Detection method: simple_baseline, sobel_flood, line_segment, or 'all' (default: simple_baseline)"
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="reports/border_detection_report.html",
        help="Output HTML report path (default: reports/border_detection_report.html)"
    )

    # Open in browser
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open report in browser after generation"
    )

    # Method parameters
    parser.add_argument(
        "--params",
        type=str,
        nargs="*",
        metavar="KEY=VALUE",
        help="Method parameters, e.g., --params threshold=50 ksize=5"
    )

    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (-v, -vv, -vvv)"
    )

    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible sampling"
    )

    # Augmentation level filtering
    parser.add_argument(
        "--aug-level",
        type=int,
        nargs="+",
        metavar="LEVEL",
        help="Filter by augmentation level (e.g., --aug-level 0 100)"
    )

    # Debug image control
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Don't include intermediate debug images in report"
    )

    return parser.parse_args()


def list_methods():
    """Print list of available detection methods."""
    from app.core.border_detection import DetectorRegistry

    print("\nAvailable detection methods:")
    print("-" * 50)

    for info in DetectorRegistry.get_info():
        print(f"\n  {info['name']}")
        print(f"    {info['description']}")
        if info['default_params']:
            print(f"    Default params:")
            for k, v in info['default_params'].items():
                print(f"      {k}: {v}")

    print()


def main():
    """Main entry point."""
    args = parse_args()

    # Handle list-methods
    if args.list_methods:
        list_methods()
        sys.exit(0)

    # Validate args
    if not args.sample and not args.images:
        print("Error: Must specify --sample N or --images PATH...")
        print("Use --help for usage information.")
        sys.exit(1)

    # Import runner (deferred to avoid import time for --help)
    from tests.border_detection.runner import TestRunner
    from tests.border_detection.report import generate_html_report

    # Create and run test
    runner = TestRunner(
        method=args.method,
        sample_size=args.sample,
        image_paths=args.images,
        output_path=args.output,
        params=args.params or [],
        verbose=args.verbose,
        seed=args.seed,
        aug_levels=args.aug_level,
    )

    print(f"Border Detection Test")
    print(f"=" * 40)
    print(f"Method: {args.method}")
    if args.sample:
        print(f"Sample size: {args.sample}")
    if args.aug_level:
        print(f"Aug levels: {args.aug_level}")
    print()

    # Run tests
    if args.method == "all":
        all_results = runner.run_comparison()
        # Generate report for first method (TODO: comparison report)
        results = list(all_results.values())[0] if all_results else None
    else:
        results = runner.run()

    if results is None or results.total == 0:
        print("No results to report.")
        sys.exit(1)

    # Generate report
    output_path = Path(args.output)
    generate_html_report(
        results,
        output_path,
        include_debug_images=not args.no_debug,
    )

    print(f"\nReport generated: {output_path}")
    print(f"Success rate: {results.success_rate:.1%} ({results.success_count}/{results.total})")
    print(f"Average time: {results.avg_time_ms:.1f}ms")

    # Open in browser if requested
    if args.open:
        webbrowser.open(f"file://{output_path.absolute()}")


if __name__ == "__main__":
    main()
