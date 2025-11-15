#!/usr/bin/env python3
"""
Test suite for TFT Level Detector using real snapshots
"""

import unittest
import cv2
import json
import os
from pathlib import Path
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(project_root)

from tft_video_analyzer.detectors.levels.detector import TFTLevelDetector


class TestLevelDetector(unittest.TestCase):
    """Test cases for the TFT Level Detector using real game snapshots"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.detector = TFTLevelDetector()

        # Use snapshots in the tests folder
        cls.snapshot_dir = Path(__file__).resolve().parent / "snapshots"

        # Load ground truth labels
        cls.ground_truth = cls.load_ground_truth()

    @classmethod
    def load_ground_truth(cls):
        """Load ground truth labels for level detection"""
        # This will be populated with your labels
        ground_truth = {
            # Format: snapshot_id: expected_level
            # null means no level should be detected
            # Example:
            # 0: 2,    # Snapshot 0 should have level 2
            # 1: 3,    # Snapshot 1 should have level 3
            # 2: None, # Snapshot 2 has no level visible
        }

        # Try to load from JSON file if it exists
        ground_truth_file = Path(__file__).parent / "level_detector_ground_truth.json"
        if ground_truth_file.exists():
            with open(ground_truth_file, 'r') as f:
                ground_truth = json.load(f)
                # Convert string keys to integers
                ground_truth = {int(k): v for k, v in ground_truth.items()}

        return ground_truth

    def test_snapshot_level_detection(self):
        """Test level detection on all snapshots with ground truth"""
        if not self.snapshot_dir.exists():
            self.skipTest(f"Snapshot directory {self.snapshot_dir} not found")

        if not self.ground_truth:
            self.skipTest("No ground truth labels available. Please run label_level_snapshots.py first")

        results = []
        errors = []

        for snapshot_id, expected_level in self.ground_truth.items():
            snapshot_path = self.snapshot_dir / f"snapshot_{snapshot_id:03d}" / "frame.png"

            if not snapshot_path.exists():
                print(f"Warning: Snapshot {snapshot_id} not found at {snapshot_path}")
                continue

            # Load the frame
            frame = cv2.imread(str(snapshot_path))

            # Detect level
            detected_level = self.detector.detect_level(frame)

            # Check if detection matches ground truth
            if detected_level == expected_level:
                results.append({
                    'snapshot_id': snapshot_id,
                    'status': 'PASS',
                    'expected': expected_level,
                    'detected': detected_level
                })
            else:
                errors.append({
                    'snapshot_id': snapshot_id,
                    'status': 'FAIL',
                    'expected': expected_level,
                    'detected': detected_level
                })

        # Print results summary
        print(f"\n{'='*50}")
        print(f"Level Detection Test Results")
        print(f"{'='*50}")
        print(f"Total tests: {len(results) + len(errors)}")
        print(f"Passed: {len(results)}")
        print(f"Failed: {len(errors)}")

        if errors:
            print(f"\nFailed tests:")
            for error in errors:
                print(f"  Snapshot {error['snapshot_id']:03d}: "
                      f"Expected {error['expected']}, Got {error['detected']}")

        # Calculate accuracy
        total = len(results) + len(errors)
        if total > 0:
            accuracy = (len(results) / total) * 100
            print(f"\nAccuracy: {accuracy:.1f}%")

        # Assert all tests passed
        self.assertEqual(len(errors), 0,
                        f"{len(errors)} tests failed. See details above.")

    


def run_tests():
    """Run the test suite"""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()
