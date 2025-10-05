#!/usr/bin/env python3
"""
Extract all player slots for debugging the health detector
"""

import cv2
import os
import sys
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(project_root)

from tft_video_analyzer.detectors.health.detector import TFTHealthDetector


def extract_all_slots_for_snapshot(snapshot_id):
    """Extract all player slots for a specific snapshot"""

    snapshot_dir = Path(__file__).parent / "snapshots" / f"snapshot_{snapshot_id:03d}"
    frame_path = snapshot_dir / "frame.png"

    if not frame_path.exists():
        print(f"Error: {frame_path} not found")
        return

    # Load frame
    frame = cv2.imread(str(frame_path))

    # Initialize detector
    detector = TFTHealthDetector()

    # Extract sidebar
    sidebar = detector.extract_sidebar_region(frame)
    cv2.imwrite(str(snapshot_dir / "health_sidebar.png"), sidebar)

    # Extract all player slots
    slots = detector.extract_player_slots(frame)

    # Detect all players
    players = detector.detect_all_players_health(frame)

    print(f"\nSnapshot {snapshot_id:03d}:")
    for (slot, slot_idx), player in zip(slots, players):
        # Save slot image
        cv2.imwrite(str(snapshot_dir / f"health_slot_{slot_idx}.png"), slot)

        # Create processed version
        gray = cv2.cvtColor(slot, cv2.COLOR_BGR2GRAY)
        scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(scaled, 150, 255, cv2.THRESH_BINARY)
        cv2.imwrite(str(snapshot_dir / f"health_slot_{slot_idx}_processed.png"), binary)

        print(f"  Slot {slot_idx}: health={player['health']}, name={player['name']}, is_user={player['is_user']}")


def extract_all():
    """Extract all player slots for all snapshots"""
    snapshot_dir = Path(__file__).parent / "snapshots"

    for snapshot_folder in sorted(snapshot_dir.glob("snapshot_*")):
        try:
            snapshot_id = int(snapshot_folder.name.split("_")[1])
            extract_all_slots_for_snapshot(snapshot_id)
        except Exception as e:
            print(f"Error processing {snapshot_folder.name}: {e}")
            continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract all player slots from test snapshots")
    parser.add_argument("--snapshot", type=int, help="Specific snapshot ID to process")
    parser.add_argument("--all", action="store_true", help="Process all snapshots")

    args = parser.parse_args()

    if args.snapshot is not None:
        extract_all_slots_for_snapshot(args.snapshot)
    elif args.all:
        extract_all()
    else:
        print("Usage:")
        print("  Extract specific snapshot: python extract_all_player_slots.py --snapshot 0")
        print("  Process all: python extract_all_player_slots.py --all")
