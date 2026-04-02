"""
Registry for border detection methods.

Provides a central registry for discovering and instantiating
detection methods by name.
"""

from typing import Dict, List, Type

from .base import BorderDetector


class DetectorRegistry:
    """
    Registry for border detection methods.

    Allows registration and lookup of detector classes by name.

    Example:
        # Register a detector
        DetectorRegistry.register(MyDetector)

        # Get detector instance
        detector = DetectorRegistry.get("my_detector")
        result = detector.detect(image)

        # List available detectors
        names = DetectorRegistry.list_all()
    """

    _detectors: Dict[str, Type[BorderDetector]] = {}

    @classmethod
    def register(cls, detector_class: Type[BorderDetector]) -> None:
        """
        Register a detector class.

        Args:
            detector_class: BorderDetector subclass to register
        """
        # Instantiate to get the name property
        instance = detector_class()
        cls._detectors[instance.name] = detector_class

    @classmethod
    def get(cls, name: str) -> BorderDetector:
        """
        Get a detector instance by name.

        Args:
            name: Detector identifier

        Returns:
            Instantiated BorderDetector

        Raises:
            ValueError: If detector not found
        """
        if name not in cls._detectors:
            available = ", ".join(sorted(cls._detectors.keys()))
            raise ValueError(f"Unknown detector: {name}. Available: {available}")
        return cls._detectors[name]()

    @classmethod
    def get_class(cls, name: str) -> Type[BorderDetector]:
        """
        Get a detector class by name (not instantiated).

        Args:
            name: Detector identifier

        Returns:
            BorderDetector subclass

        Raises:
            ValueError: If detector not found
        """
        if name not in cls._detectors:
            available = ", ".join(sorted(cls._detectors.keys()))
            raise ValueError(f"Unknown detector: {name}. Available: {available}")
        return cls._detectors[name]

    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered detector names."""
        return sorted(cls._detectors.keys())

    @classmethod
    def get_all(cls) -> List[BorderDetector]:
        """Get instances of all registered detectors."""
        return [cls._detectors[name]() for name in sorted(cls._detectors.keys())]

    @classmethod
    def get_info(cls) -> List[dict]:
        """
        Get info about all registered detectors.

        Returns:
            List of dicts with name, description, default_params
        """
        info = []
        for name in sorted(cls._detectors.keys()):
            detector = cls._detectors[name]()
            info.append({
                "name": name,
                "description": detector.description,
                "default_params": detector.get_default_params(),
            })
        return info

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a detector is registered."""
        return name in cls._detectors

    @classmethod
    def clear(cls) -> None:
        """Clear all registered detectors. Mainly for testing."""
        cls._detectors.clear()
