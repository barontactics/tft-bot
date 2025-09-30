#!/usr/bin/env python3
"""
Test suite for TFT Gold Detector using real snapshots
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

from tft_video_analyzer.detectors.gold.detector import TFTGoldDetector


class TestGoldDetector(unittest.TestCase):
    """Test cases for the TFT Gold Detector using real game snapshots"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.detector = TFTGoldDetector()

        # Use snapshots in the tests folder
        cls.snapshot_dir = Path(__file__).parent / "snapshots"

        # Load ground truth labels
        cls.ground_truth = cls.load_ground_truth()

    @classmethod
    def load_ground_truth(cls):
        """Load ground truth labels for gold detection"""
        # This will be populated with your labels
        ground_truth = {
            # Format: snapshot_id: expected_gold_amount
            # -1 means no gold should be detected
            # Example:
            # 0: 90,  # Snapshot 0 should have 90 gold
            # 1: 35,  # Snapshot 1 should have 35 gold
            # 2: -1,  # Snapshot 2 has no gold visible
        }

        # Try to load from JSON file if it exists
        ground_truth_file = Path(__file__).parent / "gold_detector_ground_truth.json"
        if ground_truth_file.exists():
            with open(ground_truth_file, 'r') as f:
                ground_truth = json.load(f)
                # Convert string keys to integers
                ground_truth = {int(k): v for k, v in ground_truth.items()}

        return ground_truth

    def test_snapshot_gold_detection(self):
        """Test gold detection on all snapshots with ground truth"""
        if not self.snapshot_dir.exists():
            self.skipTest(f"Snapshot directory {self.snapshot_dir} not found")

        if not self.ground_truth:
            self.skipTest("No ground truth labels available. Please run label_gold_snapshots.py first")

        results = []
        errors = []

        for snapshot_id, expected_gold in self.ground_truth.items():
            snapshot_path = self.snapshot_dir / f"snapshot_{snapshot_id:03d}" / "frame.png"

            if not snapshot_path.exists():
                self.skipTest(f"Snapshot {snapshot_id} not found")
                continue

            # Load the frame
            frame = cv2.imread(str(snapshot_path))

            # Detect gold
            detected_gold = self.detector.detect_gold(frame)
            if detected_gold is None:
                detected_gold = -1

            # Check if detection matches ground truth
            if detected_gold == expected_gold:
                results.append({
                    'snapshot_id': snapshot_id,
                    'status': 'PASS',
                    'expected': expected_gold,
                    'detected': detected_gold
                })
            else:
                errors.append({
                    'snapshot_id': snapshot_id,
                    'status': 'FAIL',
                    'expected': expected_gold,
                    'detected': detected_gold
                })

        # Print results summary
        print(f"\n{'='*50}")
        print(f"Gold Detection Test Results")
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

    def test_gold_detection_with_confidence(self):
        """Test gold detection with confidence scores"""
        if not self.snapshot_dir.exists():
            self.skipTest(f"Snapshot directory {self.snapshot_dir} not found")

        if not self.ground_truth:
            self.skipTest("No ground truth labels available")

        results = []

        for snapshot_id, expected_gold in self.ground_truth.items():
            snapshot_path = self.snapshot_dir / f"snapshot_{snapshot_id:03d}" / "frame.png"

            if not snapshot_path.exists():
                continue

            frame = cv2.imread(str(snapshot_path))
            detected_gold, confidence = self.detector.detect_gold_with_confidence(frame)

            if detected_gold is None:
                detected_gold = -1

            is_correct = detected_gold == expected_gold

            results.append({
                'snapshot_id': snapshot_id,
                'expected': expected_gold,
                'detected': detected_gold,
                'confidence': confidence,
                'correct': is_correct
            })

        # Analyze confidence correlation with correctness
        correct_results = [r for r in results if r['correct']]
        incorrect_results = [r for r in results if not r['correct']]

        if correct_results:
            avg_confidence_correct = sum(r['confidence'] for r in correct_results) / len(correct_results)
        else:
            avg_confidence_correct = 0

        if incorrect_results:
            avg_confidence_incorrect = sum(r['confidence'] for r in incorrect_results) / len(incorrect_results)
        else:
            avg_confidence_incorrect = 0

        print(f"\n{'='*50}")
        print(f"Confidence Analysis")
        print(f"{'='*50}")
        print(f"Average confidence for correct detections: {avg_confidence_correct:.3f}")
        print(f"Average confidence for incorrect detections: {avg_confidence_incorrect:.3f}")

        # Ideally, correct detections should have higher confidence
        if correct_results and incorrect_results:
            self.assertGreater(avg_confidence_correct, avg_confidence_incorrect,
                             "Correct detections should have higher confidence on average")

    def test_specific_problematic_cases(self):
        """Test specific cases that are known to be problematic"""
        problematic_cases = [
            # Add snapshot IDs that are known to be difficult
            # Example: 23 (detected 95, might be wrong due to icon)
        ]

        for snapshot_id in problematic_cases:
            if snapshot_id not in self.ground_truth:
                continue

            snapshot_path = self.snapshot_dir / f"snapshot_{snapshot_id:03d}" / "frame.png"
            if not snapshot_path.exists():
                continue

            frame = cv2.imread(str(snapshot_path))
            detected_gold = self.detector.detect_gold(frame)

            if detected_gold is None:
                detected_gold = -1

            expected = self.ground_truth[snapshot_id]

            print(f"\nProblematic case - Snapshot {snapshot_id}:")
            print(f"  Expected: {expected}")
            print(f"  Detected: {detected_gold}")

            # For problematic cases, we might allow some tolerance
            if expected > 0 and detected_gold > 0:
                tolerance = 5  # Allow ±5 gold difference
                self.assertAlmostEqual(detected_gold, expected, delta=tolerance,
                                     msg=f"Snapshot {snapshot_id}: Gold detection outside tolerance")


def run_tests():
    """Run the test suite"""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()