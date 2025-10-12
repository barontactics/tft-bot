#!/usr/bin/env python3
"""
Generate truth game state data by running detectors on snapshots
This creates a reference JSON file for testing video processor accuracy
"""

import cv2
import json
import sys
import easyocr
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tft_video_analyzer.detectors.round.detector import TFTRoundDetector
from tft_video_analyzer.detectors.gold.detector import TFTGoldDetector
from tft_video_analyzer.detectors.health.detector import TFTHealthDetector
from tft_video_analyzer.detectors.streak.detector import TFTStreakDetector


def generate_truth_data(snapshots_dir: str, output_file: str = "truth_game_states.json"):
    """
    Generate truth data by running all detectors on each snapshot

    Args:
        snapshots_dir: Directory containing snapshot PNG files
        output_file: Output JSON file name
    """
    snapshots_path = Path(snapshots_dir)
    if not snapshots_path.exists():
        print(f"Error: Snapshots directory not found: {snapshots_dir}")
        return

    # Get all PNG snapshots (exclude loading frames, focus on game snapshots)
    snapshot_files = sorted([f for f in snapshots_path.glob("snapshot_*_stage*.png")])

    if not snapshot_files:
        print(f"Error: No game snapshots found in {snapshots_dir}")
        return

    print(f"Found {len(snapshot_files)} game snapshots")
    print("Initializing shared EasyOCR reader...")

    # Initialize shared reader
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    # Initialize detectors with shared reader
    round_detector = TFTRoundDetector(reader=reader)
    gold_detector = TFTGoldDetector(reader=reader)
    health_detector = TFTHealthDetector(reader=reader)
    streak_detector = TFTStreakDetector(reader=reader)

    print("Detectors initialized")
    print("\nProcessing snapshots...")

    truth_data = []

    for i, snapshot_file in enumerate(snapshot_files, 1):
        # Load image
        frame = cv2.imread(str(snapshot_file))
        if frame is None:
            print(f"Warning: Could not load {snapshot_file.name}")
            continue

        # Extract metadata from filename
        # Format: snapshot_XXXX_stageX-X_frameXXXXX_XXX.Xs.png
        filename = snapshot_file.stem  # Remove .png
        parts = filename.split('_')

        snapshot_id = int(parts[1])
        frame_number = int(parts[3].replace('frame', ''))
        timestamp_str = parts[4].replace('s', '')
        timestamp = float(timestamp_str)

        # Run all detectors
        stage_round = round_detector.detect_stage_round(frame)
        gold = gold_detector.detect_gold(frame)
        health_result = health_detector.detect_health(frame)
        streak_result = streak_detector.detect_streak(frame)

        # Process streak result
        streak_value = None
        if streak_result:
            streak_type = streak_result.get('type')
            streak_length = streak_result.get('length')

            if streak_length is not None and streak_length >= 0:
                if streak_type.value == 'loss':
                    streak_value = -streak_length if streak_length > 0 else 0
                else:
                    streak_value = streak_length

        # Create truth entry
        truth_entry = {
            'snapshot_id': snapshot_id,
            'filename': snapshot_file.name,
            'frame_number': frame_number,
            'timestamp': timestamp,
            'stage': stage_round['stage'] if stage_round else None,
            'round': stage_round['round'] if stage_round else None,
            'gold': gold,
            'health': health_result if health_result else None,
            'streak': streak_value
        }

        truth_data.append(truth_entry)

        # Progress indicator
        if i % 10 == 0:
            print(f"  Processed {i}/{len(snapshot_files)} snapshots...")

    # Save truth data
    output_path = snapshots_path / output_file
    with open(output_path, 'w') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'snapshots_dir': str(snapshots_path),
            'total_snapshots': len(truth_data),
            'truth_data': truth_data
        }, f, indent=2)

    print(f"\n✓ Truth data generated: {output_path}")
    print(f"  Total entries: {len(truth_data)}")

    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate truth game state data from snapshots")
    parser.add_argument("snapshots_dir", help="Directory containing snapshot PNG files")
    parser.add_argument("-o", "--output", default="truth_game_states.json",
                       help="Output JSON filename (default: truth_game_states.json)")

    args = parser.parse_args()

    generate_truth_data(args.snapshots_dir, args.output)


if __name__ == "__main__":
    main()
