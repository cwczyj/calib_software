"""
st_handeye - Hand-eye calibration via reprojection error minimization.

Python port of st_handeye_graph, using scipy.optimize for graph optimization.
Supports pinhole camera model.
"""

from .calibrator import HandEyeCalibrator
from .types import (
    BoardConfig, CalibrationResult, CalibrationSetup, DepthSample,
    DetectionResult, OptimizationParams,
)

__all__ = [
    'HandEyeCalibrator', 'BoardConfig', 'CalibrationResult',
    'CalibrationSetup', 'DepthSample', 'OptimizationParams',
    'DetectionResult',
]
