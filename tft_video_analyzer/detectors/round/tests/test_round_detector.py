#!/usr/bin/env python3
"""
Test suite for TFT Round Detector using real snapshots
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

from tft_video_analyzer.detectors.round.detector import TFTRoundDetector


class TestRoundDetector(unittest.TestCase):
    """Test cases for the TFT Round Detector using real game snapshots"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.detector = TFTRoundDetector()

        # Use snapshots in the tests folder
        cls.snapshot_dir = Path(__file__).parent / "snapshots"

        # Load ground truth labels
        cls.ground_truth = cls.load_ground_truth()

    @classmethod
    def load_ground_truth(cls):
        """Load ground truth labels for round detection"""
        # This will be populated with your labels
        ground_truth = {
            # Format: snapshot_id: {"stage": int, "round": int}
            # None means no round should be detected
            # Example:
            # 0: {"stage": 1, "round": 2},  # Snapshot 0 should have stage 1, round 2
            # 1: {"stage": 2, "round": 3},  # Snapshot 1 should have stage 2, round 3
            # 2: None,                      # Snapshot 2 has no round visible
        }

        # Try to load from JSON file if it exists
        ground_truth_file = Path(__file__).parent / "round_detector_ground_truth.json"
        if ground_truth_file.exists():
            with open(ground_truth_file, 'r') as f:
                data = json.load(f)
                # Convert string keys to integers
                ground_truth = {int(k): v for k, v in data.items()}

        return ground_truth

    def test_snapshot_round_detection(self):
        """Test round detection on all snapshots with ground truth"""
        if not self.snapshot_dir.exists():
            self.skipTest(f"Snapshot directory {self.snapshot_dir} not found")

        if not self.ground_truth:
            self.skipTest("No ground truth labels available")

        results = []
        errors = []

        for snapshot_id, expected_round in self.ground_truth.items():
            snapshot_path = self.snapshot_dir / f"snapshot_{snapshot_id:03d}" / "frame.png"

            if not snapshot_path.exists():
                self.skipTest(f"Snapshot {snapshot_id} not found")
                continue

            # Load the frame
            frame = cv2.imread(str(snapshot_path))

            # Detect stage and round
            detected_round = self.detector.detect_stage_round(frame)

            # Check if detection matches ground truth
            if detected_round == expected_round:
                results.append({
                    'snapshot_id': snapshot_id,
                    'status': 'PASS',
                    'expected': expected_round,
                    'detected': detected_round
                })
            else:
                errors.append({
                    'snapshot_id': snapshot_id,
                    'status': 'FAIL',
                    'expected': expected_round,
                    'detected': detected_round
                })

        # Print results summary
        print(f"\n{'='*50}")
        print(f"Round Detection Test Results")
        print(f"{'='*50}")
        print(f"Total tests: {len(results) + len(errors)}")
        print(f"Passed: {len(results)}")
        print(f"Failed: {len(errors)}")

        if errors:
            print(f"\nFailed tests:")
            for error in errors:
                exp_str = f"{error['expected']['stage']}-{error['expected']['round']}" if error['expected'] else "None"
                det_str = f"{error['detected']['stage']}-{error['detected']['round']}" if error['detected'] else "None"
                print(f"  Snapshot {error['snapshot_id']:03d}: "
                      f"Expected {exp_str}, Got {det_str}")

        # Calculate accuracy
        total = len(results) + len(errors)
        if total > 0:
            accuracy = (len(results) / total) * 100
            print(f"\nAccuracy: {accuracy:.1f}%")

        # Assert all tests passed
        self.assertEqual(len(errors), 0,
                        f"{len(errors)} tests failed. See details above.")

    def test_round_detection_consistency(self):
        """Test that round detection is consistent across multiple runs"""
        if not self.snapshot_dir.exists():
            self.skipTest(f"Snapshot directory {self.snapshot_dir} not found")

        if not self.ground_truth:
            self.skipTest("No ground truth labels available")

        # Pick first snapshot with ground truth
        snapshot_id = list(self.ground_truth.keys())[0]
        snapshot_path = self.snapshot_dir / f"snapshot_{snapshot_id:03d}" / "frame.png"

        if not snapshot_path.exists():
            self.skipTest(f"Snapshot {snapshot_id} not found")

        frame = cv2.imread(str(snapshot_path))

        # Run detection multiple times
        results = []
        for _ in range(5):
            detected = self.detector.detect_stage_round(frame)
            results.append(detected)

        # Check all results are the same
        self.assertTrue(all(r == results[0] for r in results),
                       "Round detection is not consistent across multiple runs")

    def test_specific_problematic_cases(self):
        """Test specific cases that are known to be problematic"""
        problematic_cases = [
            # Add snapshot IDs that are known to be difficult
            # Example: 23 (might have UI overlays, etc)
        ]

        for snapshot_id in problematic_cases:
            if snapshot_id not in self.ground_truth:
                continue

            snapshot_path = self.snapshot_dir / f"snapshot_{snapshot_id:03d}" / "frame.png"
            if not snapshot_path.exists():
                continue

            frame = cv2.imread(str(snapshot_path))
            detected_round = self.detector.detect_stage_round(frame)

            expected = self.ground_truth[snapshot_id]

            print(f"\nProblematic case - Snapshot {snapshot_id}:")
            exp_str = f"{expected['stage']}-{expected['round']}" if expected else "None"
            det_str = f"{detected_round['stage']}-{detected_round['round']}" if detected_round else "None"
            print(f"  Expected: {exp_str}")
            print(f"  Detected: {det_str}")

            self.assertEqual(detected_round, expected,
                           f"Snapshot {snapshot_id}: Round detection mismatch")


def run_tests():
    """Run the test suite"""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()
