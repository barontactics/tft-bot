"""TFT Detectors Module - Game state detection for TFT gameplay

This module provides detectors for various game states in Teamfight Tactics:
- Augment Selection Detection
- Loading Screen Detection
- Planning Phase Detection
- Game State Management
- Video Processing
- Gold Amount Detection
- Streak Detection (Win/Loss)
"""

from .augment import TFTAugmentDetector
from .loading import TFTLoadingScreenDetector
from .planning import TFTPlanningDetector
from .game_state import TFTGameState
from .video_processor import TFTVideoProcessor
from .gold import TFTGoldDetector
from .streak import TFTStreakDetector, StreakType

__all__ = [
    'TFTAugmentDetector',
    'TFTLoadingScreenDetector',
    'TFTPlanningDetector',
    'TFTGameState',
    'TFTVideoProcessor',
    'TFTGoldDetector',
    'TFTStreakDetector',
    'StreakType'
]

# Version info
__version__ = '1.0.0'