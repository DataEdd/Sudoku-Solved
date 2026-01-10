from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    app_name: str = "Sudoku Solver"
    debug: bool = False

    # Tesseract OCR configuration
    tesseract_config: str = "--psm 10 --oem 3 -c tessedit_char_whitelist=123456789"

    # Simulated annealing solver parameters
    initial_temp: float = 1.0
    cooling_rate: float = 0.99999
    max_iterations: int = 500000

    # Hough transform parameters (HoughLinesP - probabilistic)
    hough_threshold: int = 100
    hough_min_line_length: int = 50
    hough_max_line_gap: int = 10
    line_angle_threshold: float = 10.0  # degrees
    line_cluster_distance: float = 20.0  # pixels

    # Hough transform parameters (HoughLines - polar/standard)
    hough_polar_threshold: int = 100  # Votes needed to detect line (lower = more sensitive)
    rho_threshold: float = 30.0  # pixels - max distance between similar lines
    theta_threshold: float = 0.15  # radians (~9 degrees) - max angle between similar lines

    # Preprocessing parameters for polar Hough
    canny_low: int = 50
    canny_high: int = 150
    dilate_kernel_size: int = 3
    erode_kernel_size: int = 3  # Reduced from 5 to preserve more edges

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
