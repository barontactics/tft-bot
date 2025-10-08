#!/usr/bin/env python3
"""
Helper script to extract and visualize round regions from test snapshots
Useful for debugging when labeling ground truth
"""

import cv2
import os
import sys
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(project_root)

from tft_video_analyzer.detectors.round.detector import TFTRoundDetector


def extract_regions_for_snapshot(snapshot_id, save_regions=False):
    """Extract and display round region for a specific snapshot"""

    snapshot_dir = Path(__file__).parent / "snapshots" / f"snapshot_{snapshot_id:03d}"
    frame_path = snapshot_dir / "frame.png"

    if not frame_path.exists():
        print(f"Error: {frame_path} not found")
        return None

    # Load frame
    frame = cv2.imread(str(frame_path))

    # Initialize detector
    detector = TFTRoundDetector()

    # Extract main round region
    main_region = detector.extract_region(frame, detector.round_region)
    main_processed = detector.preprocess_for_numbers(main_region)

    # Detect stage and round (this will check all regions)
    result = detector.detect_stage_round(frame)

    # Get debug info to see which region succeeded
    debug_info = detector.get_debug_info(frame)

    print(f"Snapshot {snapshot_id:03d}:")
    if result:
        print(f"  Detected: Stage {result['stage']}, Round {result['round']} ({result['stage']}-{result['round']})")
        print(f"  Region used: {debug_info['regions_checked']}")
    else:
        print(f"  Detected: None")

    if save_regions:
        # Save main region
        cv2.imwrite(str(snapshot_dir / "round_region_main.png"), main_region)
        cv2.imwrite(str(snapshot_dir / "round_processed_main.png"), main_processed)

        # If detection succeeded from an alt region, save that too
        if result and debug_info['regions_checked']:
            region_name = debug_info['regions_checked'][0]
            if region_name == 'main':
                print(f"  Saved main region images to {snapshot_dir}")
            else:
                # Extract the alt region that worked
                alt_idx = int(region_name.split('_')[1])
                alt_region_coords = detector.alt_regions[alt_idx]
                alt_region = detector.extract_region(frame, alt_region_coords)
                alt_processed = detector.preprocess_for_numbers(alt_region)

                cv2.imwrite(str(snapshot_dir / f"round_region_{region_name}.png"), alt_region)
                cv2.imwrite(str(snapshot_dir / f"round_processed_{region_name}.png"), alt_processed)
                print(f"  Saved main + {region_name} region images to {snapshot_dir}")
        else:
            print(f"  Saved main region images to {snapshot_dir}")

    return main_region, main_processed, result


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
    """Display round region for visual inspection"""
    result = extract_regions_for_snapshot(snapshot_id, save_regions=True)

    if result:
        round_region, round_processed, detection = result

        # Scale up for better visibility
        round_region_large = cv2.resize(round_region, None, fx=8, fy=8,
                                       interpolation=cv2.INTER_NEAREST)
        round_processed_large = cv2.resize(round_processed, None, fx=4, fy=4,
                                          interpolation=cv2.INTER_NEAREST)

        # Add text with detection result
        if detection:
            text = f"Stage {detection['stage']}-{detection['round']}"
            color = (0, 255, 0)  # Green for successful detection
        else:
            text = "Not detected"
            color = (0, 0, 255)  # Red for no detection

        cv2.putText(round_region_large, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        cv2.imshow(f"Round Region - Snapshot {snapshot_id:03d} (8x zoom)", round_region_large)
        cv2.imshow(f"Processed for OCR - Snapshot {snapshot_id:03d} (4x zoom)", round_processed_large)

        print("\nPress any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def compare_with_ground_truth(snapshot_id):
    """Compare detection with ground truth"""
    import json

    snapshot_dir = Path(__file__).parent / "snapshots" / f"snapshot_{snapshot_id:03d}"
    frame_path = snapshot_dir / "frame.png"
    ground_truth_file = Path(__file__).parent / "round_detector_ground_truth.json"

    if not frame_path.exists():
        print(f"Error: {frame_path} not found")
        return

    # Load ground truth
    ground_truth = {}
    if ground_truth_file.exists():
        with open(ground_truth_file, 'r') as f:
            data = json.load(f)
            ground_truth = {int(k): v for k, v in data.items()}

    # Load frame
    frame = cv2.imread(str(frame_path))

    # Initialize detector
    detector = TFTRoundDetector()

    # Detect
    detected = detector.detect_stage_round(frame)

    print(f"\nSnapshot {snapshot_id:03d} Comparison:")
    print("-" * 50)

    if snapshot_id in ground_truth:
        expected = ground_truth[snapshot_id]
        if expected:
            print(f"  Expected: Stage {expected['stage']}-{expected['round']}")
        else:
            print(f"  Expected: None")
    else:
        print(f"  Expected: Not in ground truth")

    if detected:
        print(f"  Detected: Stage {detected['stage']}-{detected['round']}")
    else:
        print(f"  Detected: None")

    # Check if match
    if snapshot_id in ground_truth:
        if detected == ground_truth[snapshot_id]:
            print(f"  Result: ✅ MATCH")
        else:
            print(f"  Result: ❌ MISMATCH")

    # Get debug info
    debug_info = detector.get_debug_info(frame)
    print(f"\nDebug Info:")
    print(f"  Regions checked: {debug_info['regions_checked']}")
    print(f"  Text found: {debug_info['text_found']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract round regions from test snapshots")
    parser.add_argument("--snapshot", type=int, help="Specific snapshot ID to process")
    parser.add_argument("--all", action="store_true", help="Process all snapshots")
    parser.add_argument("--save", action="store_true", help="Save extracted regions")
    parser.add_argument("--display", action="store_true", help="Display regions (requires OpenCV GUI)")
    parser.add_argument("--compare", action="store_true", help="Compare with ground truth")

    args = parser.parse_args()

    if args.snapshot is not None:
        if args.compare:
            compare_with_ground_truth(args.snapshot)
        elif args.display:
            display_region(args.snapshot)
        else:
            extract_regions_for_snapshot(args.snapshot, save_regions=args.save)
    elif args.all:
        extract_all_regions(save_regions=args.save)
    else:
        print("Usage:")
        print("  Extract specific snapshot: python extract_round_regions.py --snapshot 10")
        print("  Display with GUI: python extract_round_regions.py --snapshot 10 --display")
        print("  Compare with ground truth: python extract_round_regions.py --snapshot 10 --compare")
        print("  Process all: python extract_round_regions.py --all")
        print("  Save regions: python extract_round_regions.py --snapshot 10 --save")
