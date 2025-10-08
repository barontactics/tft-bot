"""TFT Detectors Module - Game state detection for TFT gameplay

This module provides detectors for various game states in Teamfight Tactics:
- Augment Selection Detection
- Loading Screen Detection
- Planning Phase Detection
- Gold Amount Detection
- Streak Detection (Win/Loss)
- Health Detection
- Round Detection (Stage-Round)
"""

from .augment import TFTAugmentDetector
from .loading import TFTLoadingScreenDetector
from .planning import TFTPlanningDetector
from .gold import TFTGoldDetector
from .streak import TFTStreakDetector, StreakType
from .health import TFTHealthDetector
from .round import TFTRoundDetector

__all__ = [
    'TFTAugmentDetector',
    'TFTLoadingScreenDetector',
    'TFTPlanningDetector',
    'TFTGoldDetector',
    'TFTStreakDetector',
    'StreakType',
    'TFTHealthDetector',
    'TFTRoundDetector'
]

# Version info
__version__ = '1.0.0'