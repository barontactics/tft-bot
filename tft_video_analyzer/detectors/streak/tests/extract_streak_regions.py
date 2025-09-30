#!/usr/bin/env python3
"""
Helper script to extract and visualize streak regions from test snapshots
Useful for debugging when labeling ground truth
"""

import cv2
import os
import sys
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(project_root)

from tft_video_analyzer.detectors.streak.detector import TFTStreakDetector


def extract_regions_for_snapshot(snapshot_id, save_regions=False):
    """Extract and display streak region for a specific snapshot"""

    snapshot_dir = Path(__file__).parent / "snapshots" / f"snapshot_{snapshot_id:03d}"
    frame_path = snapshot_dir / "frame.png"

    if not frame_path.exists():
        print(f"Error: {frame_path} not found")
        return None

    # Load frame
    frame = cv2.imread(str(frame_path))

    # Initialize detector
    detector = TFTStreakDetector()

    # Extract streak region
    streak_region = detector.extract_streak_region(frame)

    # Detect streak type with debug masks
    debug_path = snapshot_dir / "streak_color_debug"
    streak_type = detector.detect_streak_type(streak_region, save_debug_masks=save_regions,
                                              debug_path=str(debug_path))

    # Preprocess for OCR
    streak_processed = detector.preprocess_for_ocr(streak_region)

    # Detect streak (full detection)
    streak_result = detector.detect_streak(frame)

    print(f"Snapshot {snapshot_id:03d}:")
    if streak_result is None:
        print(f"  Detected: None")
        print(f"  Confidence: 0.000")
        streak_count = None
        confidence = 0.0
    else:
        streak_type = streak_result['type'].value if hasattr(streak_result['type'], 'value') else str(streak_result['type'])
        streak_count = streak_result['length']
        confidence = streak_result.get('confidence', 0.0)
        print(f"  Type: {streak_type}")
        print(f"  Length: {streak_count}")
        print(f"  Confidence: {confidence:.3f}")

    if save_regions:
        # Save extracted regions
        cv2.imwrite(str(snapshot_dir / "streak_region_temp.png"), streak_region)
        cv2.imwrite(str(snapshot_dir / "streak_processed_temp.png"), streak_processed)
        print(f"  Saved temporary region images to {snapshot_dir}")

    return streak_region, streak_processed, streak_result


def extract_all_regions(save_regions=False):
    """Extract regions for all snapshots"""
    snapshot_dir = Path(__file__).parent / "snapshots"

    for snapshot_folder in sorted(snapshot_dir.glob("snapshot_*")):
        try:
            snapshot_id = int(snapshot_folder.name.split("_")[1])
            extract_regions_for_snapshot(snapshot_id, save_regions=save_regions)
        except:
            continue


def display_region(snapshot_id):
    """Display streak region for visual inspection"""
    result = extract_regions_for_snapshot(snapshot_id, save_regions=True)

    if result:
        streak_region, streak_processed, streak_result = result

        # Scale up for better visibility
        streak_region_large = cv2.resize(streak_region, None, fx=4, fy=4,
                                      interpolation=cv2.INTER_NEAREST)

        # Extract info from result
        if streak_result is None:
            streak_type = "None"
            streak_count = 0
            confidence = 0.0
            color = (0, 0, 255)
        else:
            streak_type = streak_result['type'].value if hasattr(streak_result['type'], 'value') else str(streak_result['type'])
            streak_count = streak_result['length']
            confidence = streak_result.get('confidence', 0.0)
            color = (0, 255, 0) if streak_type == 'win' else (0, 0, 255) if streak_type == 'loss' else (128, 128, 128)

        # Add text with detection result
        cv2.putText(streak_region_large, f"{streak_type}:{streak_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(streak_region_large, f"Conf: {confidence:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow(f"Streak Region - Snapshot {snapshot_id:03d} (4x zoom)", streak_region_large)
        cv2.imshow(f"Processed for OCR - Snapshot {snapshot_id:03d}", streak_processed)

        print("\nPress any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract streak regions from test snapshots")
    parser.add_argument("--snapshot", type=int, help="Specific snapshot ID to process")
    parser.add_argument("--all", action="store_true", help="Process all snapshots")
    parser.add_argument("--save", action="store_true", help="Save extracted regions")
    parser.add_argument("--display", action="store_true", help="Display regions (requires OpenCV GUI)")

    args = parser.parse_args()

    if args.snapshot is not None:
        if args.display:
            display_region(args.snapshot)
        else:
            extract_regions_for_snapshot(args.snapshot, save_regions=args.save)
    elif args.all:
        extract_all_regions(save_regions=args.save)
    else:
        print("Usage:")
        print("  Extract specific snapshot: python extract_streak_regions.py --snapshot 10")
        print("  Display with GUI: python extract_streak_regions.py --snapshot 10 --display")
        print("  Process all: python extract_streak_regions.py --all")
        print("  Save regions: python extract_streak_regions.py --snapshot 10 --save")