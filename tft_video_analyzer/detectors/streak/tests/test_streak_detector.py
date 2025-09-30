#!/usr/bin/env python3
"""
Test suite for TFT Streak Detector using real snapshots
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

from tft_video_analyzer.detectors.streak.detector import TFTStreakDetector


class TestStreakDetector(unittest.TestCase):
    """Test cases for the TFT Streak Detector using real game snapshots"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.detector = TFTStreakDetector()

        # Use snapshots in the tests folder
        cls.snapshot_dir = Path(__file__).parent / "snapshots"

        # Load ground truth labels
        cls.ground_truth = cls.load_ground_truth()

    @classmethod
    def load_ground_truth(cls):
        """Load ground truth labels for streak detection"""
        # This will be populated with your labels
        ground_truth = {
            # Format: snapshot_id: expected_streak_count
            # -1 means no streak should be detected
            # Example:
            # 0: 3,   # Snapshot 0 should have 3 win streak
            # 1: -3,  # Snapshot 1 should have 3 loss streak (negative)
            # 2: 0,   # Snapshot 2 has no streak
        }

        # Try to load from JSON file if it exists
        ground_truth_file = Path(__file__).parent / "streak_detector_ground_truth.json"
        if ground_truth_file.exists():
            with open(ground_truth_file, 'r') as f:
                ground_truth = json.load(f)
                # Convert string keys to integers
                ground_truth = {int(k): v for k, v in ground_truth.items()}

        return ground_truth

    def test_snapshot_streak_detection(self):
        """Test streak detection on all snapshots with ground truth"""
        if not self.snapshot_dir.exists():
            self.skipTest(f"Snapshot directory {self.snapshot_dir} not found")

        if not self.ground_truth:
            self.skipTest("No ground truth labels available. Please create streak_detector_ground_truth.json first")

        results = []
        errors = []

        for snapshot_id, expected_data in self.ground_truth.items():
            snapshot_path = self.snapshot_dir / f"snapshot_{snapshot_id:03d}" / "frame.png"

            if not snapshot_path.exists():
                self.skipTest(f"Snapshot {snapshot_id} not found")
                continue

            # Load the frame
            frame = cv2.imread(str(snapshot_path))

            # Detect streak
            detected_result = self.detector.detect_streak(frame)

            # Extract expected values
            expected_type = expected_data.get('type', '')
            expected_length = expected_data.get('length', -1)

            # Handle None result
            if detected_result is None:
                detected_type = 'none'
                detected_length = -1
            else:
                detected_type = str(detected_result['type'].value) if hasattr(detected_result['type'], 'value') else str(detected_result['type'])
                detected_length = detected_result['length']

            # Check if detection matches ground truth
            # Handle empty string type as 'none' for comparison
            if expected_type == '':
                expected_type = 'none'

            type_match = detected_type == expected_type
            length_match = detected_length == expected_length

            if type_match and length_match:
                results.append({
                    'snapshot_id': snapshot_id,
                    'status': 'PASS',
                    'expected': f"{expected_type}:{expected_length}",
                    'detected': f"{detected_type}:{detected_length}"
                })
            else:
                errors.append({
                    'snapshot_id': snapshot_id,
                    'status': 'FAIL',
                    'expected': f"{expected_type}:{expected_length}",
                    'detected': f"{detected_type}:{detected_length}",
                    'type_match': type_match,
                    'length_match': length_match
                })

        # Print results summary
        print(f"\n{'='*50}")
        print(f"Streak Detection Test Results")
        print(f"{'='*50}")
        print(f"Total tests: {len(results) + len(errors)}")
        print(f"Passed: {len(results)}")
        print(f"Failed: {len(errors)}")

        if errors:
            print(f"\nFailed tests:")
            for error in errors:
                match_info = []
                if not error['type_match']:
                    match_info.append("type mismatch")
                if not error['length_match']:
                    match_info.append("length mismatch")
                print(f"  Snapshot {error['snapshot_id']:03d}: "
                      f"Expected {error['expected']}, Got {error['detected']} ({', '.join(match_info)})")

        # Calculate accuracy
        total = len(results) + len(errors)
        if total > 0:
            accuracy = (len(results) / total) * 100
            print(f"\nAccuracy: {accuracy:.1f}%")

        # Assert all tests passed
        self.assertEqual(len(errors), 0,
                        f"{len(errors)} tests failed. See details above.")

    def test_streak_detection_with_confidence(self):
        """Test streak detection with confidence scores"""
        if not self.snapshot_dir.exists():
            self.skipTest(f"Snapshot directory {self.snapshot_dir} not found")

        if not self.ground_truth:
            self.skipTest("No ground truth labels available")

        results = []

        for snapshot_id, expected_data in self.ground_truth.items():
            snapshot_path = self.snapshot_dir / f"snapshot_{snapshot_id:03d}" / "frame.png"

            if not snapshot_path.exists():
                continue

            frame = cv2.imread(str(snapshot_path))
            detected_result = self.detector.detect_streak(frame)

            # Extract expected values
            expected_type = expected_data.get('type', '')
            expected_length = expected_data.get('length', -1)
            if expected_type == '':
                expected_type = 'none'

            # Handle detection result
            if detected_result is None:
                detected_type = 'none'
                detected_length = -1
                confidence = 0.0
            else:
                detected_type = str(detected_result['type'].value) if hasattr(detected_result['type'], 'value') else str(detected_result['type'])
                detected_length = detected_result['length']
                confidence = detected_result.get('confidence', 0.0)

            is_correct = (detected_type == expected_type and detected_length == expected_length)

            results.append({
                'snapshot_id': snapshot_id,
                'expected_type': expected_type,
                'expected_length': expected_length,
                'detected_type': detected_type,
                'detected_length': detected_length,
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
        print(f"Correct detections: {len(correct_results)}/{len(results)}")
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
            # Example: cases with overlapping text, partial occlusion, etc.
        ]

        for snapshot_id in problematic_cases:
            if snapshot_id not in self.ground_truth:
                continue

            snapshot_path = self.snapshot_dir / f"snapshot_{snapshot_id:03d}" / "frame.png"
            if not snapshot_path.exists():
                continue

            frame = cv2.imread(str(snapshot_path))
            detected_result = self.detector.detect_streak(frame)

            expected_data = self.ground_truth[snapshot_id]
            expected_type = expected_data.get('type', '')
            expected_length = expected_data.get('length', -1)
            if expected_type == '':
                expected_type = 'none'

            if detected_result is None:
                detected_type = 'none'
                detected_length = -1
            else:
                detected_type = str(detected_result['type'].value) if hasattr(detected_result['type'], 'value') else str(detected_result['type'])
                detected_length = detected_result['length']

            print(f"\nProblematic case - Snapshot {snapshot_id}:")
            print(f"  Expected: {expected_type}:{expected_length}")
            print(f"  Detected: {detected_type}:{detected_length}")

            # For problematic cases, we expect exact match
            self.assertEqual(detected_type, expected_type,
                           msg=f"Snapshot {snapshot_id}: Streak type mismatch")
            self.assertEqual(detected_length, expected_length,
                           msg=f"Snapshot {snapshot_id}: Streak length mismatch")


def run_tests():
    """Run the test suite"""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()