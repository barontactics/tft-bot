# Test Snapshots for Gold Detector

This directory contains 33 snapshots extracted from TFT gameplay video for testing the gold detector.

## Source
- **Video**: League of Legends_09-20-2025_15-28-29-883.mp4
- **Duration**: ~28 minutes (1727.87 seconds)
- **Extraction Date**: September 29, 2025
- **Analysis ID**: video_analysis_20250929_111848

## Contents

### Snapshot Folders (snapshot_000 to snapshot_032)
Each snapshot folder contains:
- `frame.png` - Full 1920x1080 game frame
- `detection_results.json` - Detection results for this frame

Note: Processing images (regions, visualizations) have been removed to save space.
To regenerate them, run the analysis on the frame.png files.

### Combined Results
- `combined_results.json` - All detection results from the original analysis

## Frame Distribution
Frames were randomly sampled across the entire video:
- Earliest frame: 1590 (26.5 seconds)
- Latest frame: 100853 (1680.88 seconds)
- Coverage: Full game session including various game states

## Current Detection Results

| Snapshot | Frame # | Gold Detected | Confidence |
|----------|---------|---------------|------------|
| 000 | 5047 | 90 | 0.273 |
| 001 | 6716 | 35 | 0.092 |
| 002 | 9002 | -1 | 0.000 |
| 003 | 10527 | 1 | 0.094 |
| 004 | 14914 | 8 | 0.558 |
| 005 | 17647 | 15 | 0.110 |
| 006 | 21361 | 12 | 0.398 |
| 007 | 24804 | 19 | 0.375 |
| 008 | 30173 | 28 | 0.320 |
| 009 | 34173 | 30 | 0.321 |
| 010 | 34555 | 37 | 0.093 |
| 011 | 38002 | 40 | 0.093 |
| 012 | 46176 | 50 | 0.385 |
| 013 | 46394 | 50 | 0.303 |
| 014 | 47293 | 51 | 0.092 |
| 015 | 50665 | 58 | 0.391 |
| 016 | 50853 | 6 | 0.092 |
| 017 | 51632 | 6 | 0.092 |
| 018 | 52050 | 61 | 0.273 |
| 019 | 58259 | 50 | 0.372 |
| 020 | 60945 | 56 | 0.381 |
| 021 | 63101 | 21 | 0.270 |
| 022 | 65487 | 51 | 0.092 |
| 023 | 66298 | 95 | 0.093 |
| 024 | 67534 | 2 | 0.093 |
| 025 | 81487 | 12 | 0.350 |
| 026 | 81869 | -1 | 0.000 |
| 027 | 83051 | 10 | 0.208 |
| 028 | 88528 | 19 | 0.375 |
| 029 | 89169 | -1 | 0.000 |
| 030 | 97403 | 30 | 0.311 |
| 031 | 98473 | 2 | 0.084 |
| 032 | 100853 | -1 | 0.000 |

## Usage
These snapshots are used for:
1. Testing gold detection accuracy
2. Validating confidence scores
3. Regression testing after detector improvements
4. Creating ground truth labels for training