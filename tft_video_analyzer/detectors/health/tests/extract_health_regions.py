#!/usr/bin/env python3
"""
Helper script to extract and visualize health regions from test snapshots
Useful for debugging when labeling ground truth
"""

import cv2
import os
import sys
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(project_root)

from tft_video_analyzer.detectors.health.detector import TFTHealthDetector


def extract_regions_for_snapshot(snapshot_id, save_regions=False):
    """Extract and display health region for a specific snapshot"""

    snapshot_dir = Path(__file__).parent / "snapshots" / f"snapshot_{snapshot_id:03d}"
    frame_path = snapshot_dir / "frame.png"

    if not frame_path.exists():
        print(f"Error: {frame_path} not found")
        return None

    # Load frame
    frame = cv2.imread(str(frame_path))

    # Initialize detector
    detector = TFTHealthDetector()

    # Extract health region
    result = detector.extract_health_region(frame)

    # Detect health (full detection)
    health_result = detector.detect_health(frame)

    print(f"Snapshot {snapshot_id:03d}:")
    if health_result.get('health') is None:
        print(f"  Detected: None")
        print(f"  Confidence: {health_result.get('confidence', 0.0):.3f}")
        health_value = None
        confidence = health_result.get('confidence', 0.0)
    else:
        health_value = health_result['health']
        confidence = health_result.get('confidence', 0.0)
        print(f"  Health: {health_value}")
        print(f"  Confidence: {confidence:.3f}")

    if save_regions and result is not None:
        health_region, info = result
        # Save extracted region
        cv2.imwrite(str(snapshot_dir / "health_region_temp.png"), health_region)

        # Save processed version
        health_processed = detector.preprocess_for_ocr(health_region)
        cv2.imwrite(str(snapshot_dir / "health_processed_temp.png"), health_processed)

        # Save visualization
        vis_frame = detector.visualize_health_detection(frame)
        cv2.imwrite(str(snapshot_dir / "health_visualization_temp.png"), vis_frame)

        print(f"  Saved temporary region images to {snapshot_dir}")

    return result, health_result


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
    """Display health region for visual inspection"""
    result = extract_regions_for_snapshot(snapshot_id, save_regions=True)

    if result:
        region_result, health_result = result

        if region_result is None:
            print("No health region found (user player not detected)")
            return

        health_region, info = region_result

        # Scale up for better visibility
        health_region_large = cv2.resize(health_region, None, fx=4, fy=4,
                                      interpolation=cv2.INTER_NEAREST)

        # Extract info from result
        health_value = health_result.get('health')
        confidence = health_result.get('confidence', 0.0)

        if health_value is None:
            color = (0, 0, 255)
            label = "None"
        else:
            color = (0, 255, 0)
            label = str(health_value)

        # Add text with detection result
        cv2.putText(health_region_large, f"Health: {label}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(health_region_large, f"Conf: {confidence:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow(f"Health Region - Snapshot {snapshot_id:03d} (4x zoom)", health_region_large)

        # Also show processed version
        health_processed = detector.preprocess_for_ocr(health_region)
        cv2.imshow(f"Processed for OCR - Snapshot {snapshot_id:03d}", health_processed)

        print("\nPress any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract health regions from test snapshots")
    parser.add_argument("--snapshot", type=int, help="Specific snapshot ID to process")
    parser.add_argument("--all", action="store_true", help="Process all snapshots")
    parser.add_argument("--save", action="store_true", help="Save extracted regions")
    parser.add_argument("--display", action="store_true", help="Display regions (requires OpenCV GUI)")

    args = parser.parse_args()

    # Initialize detector for display mode
    if args.display:
        detector = TFTHealthDetector()

    if args.snapshot is not None:
        if args.display:
            display_region(args.snapshot)
        else:
            extract_regions_for_snapshot(args.snapshot, save_regions=args.save)
    elif args.all:
        extract_all_regions(save_regions=args.save)
    else:
        print("Usage:")
        print("  Extract specific snapshot: python extract_health_regions.py --snapshot 10")
        print("  Display with GUI: python extract_health_regions.py --snapshot 10 --display")
        print("  Process all: python extract_health_regions.py --all")
        print("  Save regions: python extract_health_regions.py --snapshot 10 --save")