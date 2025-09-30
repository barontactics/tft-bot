# TFT Streak Detector Test Suite

This directory contains test snapshots and utilities for testing the TFT Streak Detector.

## Directory Structure

```
tests/
├── __init__.py
├── README.md
├── test_streak_detector.py           # Main test suite
├── extract_streak_regions.py         # Helper to visualize streak regions
├── streak_detector_ground_truth.json # Ground truth labels for test snapshots
└── snapshots/                         # Test snapshot images
    ├── snapshot_000/
    │   └── frame.png
    ├── snapshot_001/
    │   └── frame.png
    └── ...
```

## Running Tests

### Run the full test suite:
```bash
pytest tft_video_analyzer/detectors/streak/tests/test_streak_detector.py -v
```

### Run specific test:
```bash
pytest tft_video_analyzer/detectors/streak/tests/test_streak_detector.py::TestStreakDetector::test_snapshot_streak_detection -v
```

## Creating Ground Truth Labels

1. Create or update `streak_detector_ground_truth.json` with the correct streak values for each snapshot:
```json
{
  "0": 3,    // Win streak of 3
  "1": -2,   // Loss streak of 2 (negative)
  "2": 0,    // No streak
  "3": 5     // Win streak of 5
}
```

Format:
- Positive numbers: Win streaks
- Negative numbers: Loss streaks
- 0: No streak
- Use snapshot ID (number) as key

## Extracting and Visualizing Regions

### Extract streak region for a specific snapshot:
```bash
python tft_video_analyzer/detectors/streak/tests/extract_streak_regions.py --snapshot 10
```

### Extract and save regions for all snapshots:
```bash
python tft_video_analyzer/detectors/streak/tests/extract_streak_regions.py --all --save
```

### Display region with GUI (for visual inspection):
```bash
python tft_video_analyzer/detectors/streak/tests/extract_streak_regions.py --snapshot 10 --display
```

This creates:
- `streak_region_temp.png` - The extracted streak region from the frame
- `streak_processed_temp.png` - The preprocessed version used for OCR

## Adding New Test Snapshots

1. Add a new snapshot folder: `snapshots/snapshot_XXX/`
2. Add the frame image: `snapshot_XXX/frame.png`
3. Extract the region to verify detection:
   ```bash
   python extract_streak_regions.py --snapshot XXX --save
   ```
4. Add the ground truth label to `streak_detector_ground_truth.json`
5. Run tests to verify

## Test Results

The test suite measures:
- **Accuracy**: Percentage of correctly detected streaks
- **Confidence correlation**: Whether correct detections have higher confidence scores
- **Problematic cases**: Specific snapshots known to be challenging

Example output:
```
==================================================
Streak Detection Test Results
==================================================
Total tests: 25
Passed: 23
Failed: 2

Failed tests:
  Snapshot 004: Expected 3, Got 0
  Snapshot 015: Expected -2, Got 0

Accuracy: 92.0%
```

## Current Test Coverage

- **Total snapshots**: TBD
- **Accuracy**: TBD%
- **Known issues**: TBD

## Notes

- Streak detection is more challenging than gold detection due to:
  - Smaller text and icons
  - Win/loss indicators that need interpretation
  - Variable positioning depending on streak type
  - Overlapping UI elements in some game states