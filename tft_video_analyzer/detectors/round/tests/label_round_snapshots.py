#!/usr/bin/env python3
"""
Script to label round snapshots with ground truth stage-round values
"""

import cv2
import json
import os
from pathlib import Path
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(project_root)

from tft_video_analyzer.detectors.round.detector import TFTRoundDetector


def auto_label_snapshots():
    """Automatically label snapshots using the round detector"""

    snapshot_dir = Path(__file__).parent / "snapshots"
    output_file = Path(__file__).parent / "round_detector_ground_truth.json"

    if not snapshot_dir.exists():
        print(f"Error: Snapshot directory not found: {snapshot_dir}")
        return

    print("Auto-labeling snapshots with round detector...")
    print(f"Snapshot directory: {snapshot_dir}")
    print("-" * 60)

    # Initialize detector
    detector = TFTRoundDetector()

    # Collect all snapshots
    snapshots = sorted([d for d in snapshot_dir.iterdir() if d.is_dir() and d.name.startswith("snapshot_")])

    ground_truth = {}

    for snapshot_dir_path in snapshots:
        # Extract snapshot ID from directory name
        snapshot_id = int(snapshot_dir_path.name.split("_")[1])

        # Load frame
        frame_path = snapshot_dir_path / "frame.png"
        if not frame_path.exists():
            print(f"  Snapshot {snapshot_id:03d}: No frame.png found, skipping")
            continue

        frame = cv2.imread(str(frame_path))

        # Detect round
        detected = detector.detect_stage_round(frame)

        if detected:
            ground_truth[snapshot_id] = detected
            print(f"  Snapshot {snapshot_id:03d}: Detected {detected['stage']}-{detected['round']}")
        else:
            ground_truth[snapshot_id] = None
            print(f"  Snapshot {snapshot_id:03d}: No round detected")

    # Save ground truth
    with open(output_file, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Ground truth saved to: {output_file}")
    print(f"Total snapshots labeled: {len(ground_truth)}")
    detected_count = sum(1 for v in ground_truth.values() if v is not None)
    print(f"Snapshots with rounds detected: {detected_count}")
    print(f"Snapshots without rounds: {len(ground_truth) - detected_count}")


def manual_review():
    """Manually review and correct the auto-labeled ground truth"""

    snapshot_dir = Path(__file__).parent / "snapshots"
    ground_truth_file = Path(__file__).parent / "round_detector_ground_truth.json"

    if not ground_truth_file.exists():
        print("Error: Ground truth file not found. Run auto-labeling first.")
        return

    # Load existing ground truth
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)

    print("Manual review mode")
    print("For each snapshot, verify the detected round or enter correct value")
    print("Commands: (stage-round), 'n' for none, 's' to skip, 'q' to quit")
    print("-" * 60)

    for snapshot_id_str, detected in ground_truth.items():
        snapshot_id = int(snapshot_id_str)
        snapshot_path = snapshot_dir / f"snapshot_{snapshot_id:03d}" / "frame.png"

        if not snapshot_path.exists():
            continue

        # Display current detection
        if detected:
            print(f"\nSnapshot {snapshot_id:03d}: Detected {detected['stage']}-{detected['round']}")
        else:
            print(f"\nSnapshot {snapshot_id:03d}: No round detected")

        # Get user input
        response = input("  Correct? [y/stage-round/n/s/q]: ").strip().lower()

        if response == 'q':
            break
        elif response == 's' or response == 'y':
            continue
        elif response == 'n':
            ground_truth[snapshot_id_str] = None
            print("  Updated to: None")
        elif '-' in response:
            try:
                parts = response.split('-')
                stage = int(parts[0])
                round_num = int(parts[1])
                ground_truth[snapshot_id_str] = {"stage": stage, "round": round_num}
                print(f"  Updated to: {stage}-{round_num}")
            except:
                print("  Invalid format, skipping")

    # Save updated ground truth
    with open(ground_truth_file, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\nGround truth saved to: {ground_truth_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Label round snapshots")
    parser.add_argument("--review", action="store_true", help="Manual review mode")

    args = parser.parse_args()

    if args.review:
        manual_review()
    else:
        auto_label_snapshots()
