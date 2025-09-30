#!/usr/bin/env python3
"""
Helper script to extract and visualize gold regions from test snapshots
Useful for debugging when labeling ground truth
"""

import cv2
import os
import sys
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(project_root)

from tft_video_analyzer.detectors.gold.detector import TFTGoldDetector


def extract_regions_for_snapshot(snapshot_id, save_regions=False):
    """Extract and display gold region for a specific snapshot"""

    snapshot_dir = Path(__file__).parent / "snapshots" / f"snapshot_{snapshot_id:03d}"
    frame_path = snapshot_dir / "frame.png"

    if not frame_path.exists():
        print(f"Error: {frame_path} not found")
        return None

    # Load frame
    frame = cv2.imread(str(frame_path))

    # Initialize detector
    detector = TFTGoldDetector()

    # Extract gold region
    gold_region = detector.extract_gold_region(frame)

    # Preprocess for OCR
    gold_processed = detector.preprocess_for_ocr(gold_region)

    # Detect gold
    gold_amount, confidence = detector.detect_gold_with_confidence(frame)

    print(f"Snapshot {snapshot_id:03d}:")
    print(f"  Detected: {gold_amount if gold_amount is not None else 'None'}")
    print(f"  Confidence: {confidence:.3f}")

    if save_regions:
        # Save extracted regions
        cv2.imwrite(str(snapshot_dir / "gold_region_temp.png"), gold_region)
        cv2.imwrite(str(snapshot_dir / "gold_processed_temp.png"), gold_processed)
        print(f"  Saved temporary region images to {snapshot_dir}")

    return gold_region, gold_processed, gold_amount, confidence


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
    """Display gold region for visual inspection"""
    result = extract_regions_for_snapshot(snapshot_id, save_regions=True)

    if result:
        gold_region, gold_processed, gold_amount, confidence = result

        # Scale up for better visibility
        gold_region_large = cv2.resize(gold_region, None, fx=4, fy=4,
                                      interpolation=cv2.INTER_NEAREST)

        # Add text with detection result
        cv2.putText(gold_region_large, f"Gold: {gold_amount}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 215, 0), 2)
        cv2.putText(gold_region_large, f"Conf: {confidence:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 215, 0), 2)

        cv2.imshow(f"Gold Region - Snapshot {snapshot_id:03d} (4x zoom)", gold_region_large)
        cv2.imshow(f"Processed for OCR - Snapshot {snapshot_id:03d}", gold_processed)

        print("\nPress any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract gold regions from test snapshots")
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
        print("  Extract specific snapshot: python extract_gold_regions.py --snapshot 10")
        print("  Display with GUI: python extract_gold_regions.py --snapshot 10 --display")
        print("  Process all: python extract_gold_regions.py --all")
        print("  Save regions: python extract_gold_regions.py --snapshot 10 --save")