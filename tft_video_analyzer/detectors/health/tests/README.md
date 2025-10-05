# TFT Health Detector Tests

This directory contains test snapshots and test cases for the TFT Health Detector.

## Structure

```
tests/
├── snapshots/               # Test snapshots organized by ID
│   ├── snapshot_000/
│   │   └── frame.png
│   ├── snapshot_001/
│   │   └── frame.png
│   └── ...
├── health_detector_ground_truth.json  # Expected health values for each snapshot
├── test_health_detector.py            # Main test suite
├── extract_health_regions.py          # Helper script for debugging
└── README.md                           # This file
```

## Ground Truth Format

The `health_detector_ground_truth.json` file contains expected health values:

```json
{
  "0": -1,    // Snapshot 0: No health detected
  "1": -1,    // Snapshot 1: No health detected
  "27": 4,    // Snapshot 27: Health value is 4
  ...
}
```

Where:
- `-1` indicates no health should be detected (user player not found)
- `1-100` indicates the expected health value for the user's player

## Running Tests

### Run all tests
```bash
cd /Users/fridley/Documents/tft-bot
python -m pytest tft_video_analyzer/detectors/health/tests/test_health_detector.py -v
```

Or using unittest:
```bash
python tft_video_analyzer/detectors/health/tests/test_health_detector.py
```

### Run specific test
```bash
python -m pytest tft_video_analyzer/detectors/health/tests/test_health_detector.py::TestHealthDetector::test_snapshot_health_detection -v
```

## Debugging Tools

### Extract and visualize health regions

Extract health region for a specific snapshot:
```bash
python tft_video_analyzer/detectors/health/tests/extract_health_regions.py --snapshot 27
```

Display with GUI (requires display):
```bash
python tft_video_analyzer/detectors/health/tests/extract_health_regions.py --snapshot 27 --display
```

Process all snapshots:
```bash
python tft_video_analyzer/detectors/health/tests/extract_health_regions.py --all
```

Save extracted regions:
```bash
python tft_video_analyzer/detectors/health/tests/extract_health_regions.py --snapshot 27 --save
```

This will create temporary debug images in the snapshot directory:
- `health_region_temp.png` - Extracted health region
- `health_processed_temp.png` - Preprocessed for OCR
- `health_visualization_temp.png` - Full frame with detection overlay

## Adding New Test Cases

1. Add snapshots to `snapshots/snapshot_XXX/frame.png`
2. Run the detector to get the detected value
3. Manually verify the correct health value from the frame
4. Add the entry to `health_detector_ground_truth.json`:
   ```json
   {
     ...
     "XXX": expected_health_value
   }
   ```
5. Run tests to verify

## Test Coverage

The test suite includes:
1. **Basic Detection Test**: Validates detection against ground truth for all snapshots
2. **Confidence Analysis**: Analyzes correlation between confidence scores and correctness
3. **Problematic Cases**: Tests specific difficult cases

## Notes

- Health values range from 1-100 in TFT
- The detector identifies the user's player by detecting the yellow circle around their icon
- Detection failures typically occur when:
  - User player is not visible (eliminated)
  - Yellow circle is occluded or not visible
  - OCR fails to read the health number
  - Health number is too faint or blurred