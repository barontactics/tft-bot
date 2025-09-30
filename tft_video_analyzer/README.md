# TFT Video Analyzer

Automatically analyze TeamFight Tactics gameplay videos and capture snapshots at critical moments.

## Features

- **Automatic Critical Moment Detection**
  - Carousel rounds
  - Combat phases
  - Round transitions
  - Low health situations
  - High gold moments
  - Major damage events

- **Smart Frame Analysis**
  - Game state detection using computer vision
  - OCR for health/gold extraction
  - Scene change detection
  - State transition tracking

- **Organized Output**
  - Categorized snapshots (carousel, combat, transitions, critical)
  - Metadata tracking for each snapshot
  - Summary image generation
  - JSON metadata export

## Installation

```bash
cd tft_video_analyzer
pip install -r requirements.txt
```

### Additional Requirements

For OCR functionality (reading health/gold):
```bash
# macOS
brew install tesseract

# Linux
sudo apt-get install tesseract-ocr
```

## Usage

### Basic Usage
```bash
python analyze_tft_video.py path/to/video.mp4
```

### Advanced Options
```bash
# Customize output directory
python analyze_tft_video.py video.mp4 -o custom_output_dir

# Adjust sampling rate (check every 0.25 seconds)
python analyze_tft_video.py video.mp4 -r 0.25

# Limit maximum snapshots
python analyze_tft_video.py video.mp4 -m 50

# Aggressive mode (captures more moments)
python analyze_tft_video.py video.mp4 -a
```

## Project Structure

```
tft_video_analyzer/
├── analyze_tft_video.py      # Main script
├── video_processor.py         # Video handling
├── tft_detector.py           # Game state detection
├── moment_detector.py        # Critical moment identification
├── snapshot_manager.py       # Snapshot organization
├── config/
│   └── detection_config.json # Detection parameters
├── snapshots/                # Output directory
├── videos/                   # Input videos
└── models/                   # Future ML models
```

## Output Structure

Each analysis session creates:
```
snapshots/
└── VideoName_YYYYMMDD_HHMMSS/
    ├── carousel/            # Carousel round snapshots
    ├── combat/              # Combat phase snapshots
    ├── transitions/         # Round transitions
    ├── critical/            # Critical moments (low health, etc)
    ├── misc/               # Other moments
    ├── metadata.json       # Session metadata
    └── summary.jpg         # Grid of key moments
```

## Configuration

Edit `config/detection_config.json` to adjust:
- Screen region coordinates
- Detection thresholds
- Priority levels for different events
- Capture settings

## How It Works

1. **Video Loading**: Opens video file and extracts basic info
2. **Frame Sampling**: Processes frames at specified intervals
3. **State Detection**: Analyzes each frame for game state
4. **Moment Detection**: Identifies critical moments based on state changes
5. **Snapshot Capture**: Saves important frames with metadata
6. **Report Generation**: Creates summary and statistics

## Tips for Best Results

- Use 1080p or higher resolution videos
- Ensure UI elements are clearly visible
- Adjust region coordinates if using different resolutions
- Use aggressive mode for more comprehensive capture
- Lower sample rate for more precise moment detection

## Future Improvements

- Machine learning models for champion detection
- Item combination tracking
- Board state analysis
- Automatic highlight reel generation
- Real-time stream analysis support