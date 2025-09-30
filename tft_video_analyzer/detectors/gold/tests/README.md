# Gold Detector Tests

This directory contains tests and utilities for the TFT Gold Detector.

## Files

- `test_gold_detector.py` - Main test suite that validates detection accuracy
- `quick_label_gold.py` - Quick labeling tool for creating ground truth labels
- `label_gold_snapshots.py` - Visual labeling tool with OpenCV display
- `gold_detector_ground_truth.json` - Ground truth labels (created after labeling)

## Usage

### Step 1: Create Ground Truth Labels

First, you need to label the correct gold amounts for test snapshots.

### Step 2: Run Tests

Once you have labeled the snapshots:

```bash
# From project root:
python tft_video_analyzer/detectors/gold/tests/test_gold_detector.py
```

Or run as a module:
```bash
python -m tft_video_analyzer.detectors.gold.tests.test_gold_detector
```

## Test Data

The tests use snapshots stored in the `snapshots/` subdirectory.
These are copies of the original analysis from `video_analysis_20250929_111848/`.

Each snapshot contains:
- `frame.png` - Full game frame

## Ground Truth Format

The ground truth is stored in JSON format:
```json
{
  "0": 90,   // Snapshot 0 has 90 gold
  "1": 35,   // Snapshot 1 has 35 gold
  "2": -1,   // Snapshot 2 has no gold visible
  ...
}
```

## Adding New Test Cases

To add new test snapshots:
1. Run video analysis to generate new snapshots
2. Update the snapshot directory path in test files
3. Run the labeling tool to create ground truth
4. Run tests to validate