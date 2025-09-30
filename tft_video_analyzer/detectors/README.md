# TFT Detectors

Modular detection system for Teamfight Tactics game states.

## Structure

```
detectors/
├── augment/           # Augment selection detection
│   ├── __init__.py
│   └── detector.py    # Detects 3-card augment selection screens
├── loading/           # Loading screen detection
│   ├── __init__.py
│   └── detector.py    # Detects 8-card champion loading screens
├── planning/          # Planning phase detection
│   ├── __init__.py
│   └── detector.py    # Detects yellow timer for planning phase
├── game_state/        # Game state management
│   ├── __init__.py
│   └── manager.py     # Tracks overall game state transitions
└── video_processor/   # Video processing pipeline
    ├── __init__.py
    └── processor.py   # Processes videos and captures key moments
```

## Usage

```python
from detectors import (
    TFTAugmentDetector,
    TFTLoadingScreenDetector,
    TFTPlanningDetector,
    TFTGameState,
    TFTVideoProcessor
)

# Initialize detectors
augment_detector = TFTAugmentDetector()
loading_detector = TFTLoadingScreenDetector()
planning_detector = TFTPlanningDetector()

# Process a frame
frame = cv2.imread("screenshot.png")

if augment_detector.detect_augment(frame):
    print("Augment selection detected!")
elif loading_detector.detect_loading_screen(frame):
    print("Loading screen detected!")
elif planning_detector.detect_planning_phase(frame):
    print("Planning phase detected!")
```

## Detection Criteria

### Augment Selection
- 20-35% purple content
- "Choose One" text present (10-30% white pixels in text region)
- Exactly 3 purple cards in specific positions
- Not a loading or transition screen

### Loading Screen
- >40% purple content (8 champion cards)
- No "Choose One" text pattern
- Cards spread across the screen

### Planning Phase
- Yellow timer in top-left corner (9.5% from left, 4.8% from top)
- >10% yellow pixels in timer region

### Transition Screen
- >80% black pixels
- Used to filter out scene changes

### Combat Detection
- Red or green health bars in middle region
- Not a shop screen

### Shop Detection
- Gold UI elements in bottom 25% of screen
- High edge density in bottom region