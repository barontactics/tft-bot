#!/usr/bin/env python3
"""
Test suite for TFT Health Detector using real snapshots
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

from tft_video_analyzer.detectors.health.detector import TFTHealthDetector


class TestHealthDetector(unittest.TestCase):
    """Test cases for the TFT Health Detector using real game snapshots"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.detector = TFTHealthDetector()

        # Use snapshots in the tests folder
        cls.snapshot_dir = Path(__file__).parent / "snapshots"

        # Load ground truth labels
        cls.ground_truth = cls.load_ground_truth()

    @classmethod
    def load_ground_truth(cls):
        """Load ground truth labels for health detection"""
        ground_truth = {}

        # Try to load from JSON file if it exists
        ground_truth_file = Path(__file__).parent / "health_detector_ground_truth.json"
        if ground_truth_file.exists():
            with open(ground_truth_file, 'r') as f:
                ground_truth = json.load(f)
                # Convert string keys to integers
                ground_truth = {int(k): v for k, v in ground_truth.items()}

        return ground_truth

    def test_snapshot_health_detection(self):
        """Test health detection on all snapshots with ground truth"""
        if not self.snapshot_dir.exists():
            self.skipTest(f"Snapshot directory {self.snapshot_dir} not found")

        if not self.ground_truth:
            self.skipTest("No ground truth labels available. Please create health_detector_ground_truth.json first")

        results = []
        errors = []

        for snapshot_id, expected_data in self.ground_truth.items():
            snapshot_path = self.snapshot_dir / f"snapshot_{snapshot_id:03d}" / "frame.png"

            if not snapshot_path.exists():
                continue

            # Load the frame
            frame = cv2.imread(str(snapshot_path))

            # Detect all players
            detected_players = self.detector.detect_all_players_health(frame)

            # Get expected players
            expected_players = expected_data['all_players']

            # Check if all players match (comparing only health, name, is_user)
            all_match = True
            for detected, expected in zip(detected_players, expected_players):
                det_health = detected.get('health')
                det_name = detected.get('name')
                det_is_user = detected.get('is_user')

                if (det_health != expected['health'] or
                    det_name != expected['name'] or
                    det_is_user != expected['is_user']):
                    all_match = False
                    break

            if all_match:
                results.append({
                    'snapshot_id': snapshot_id,
                    'status': 'PASS'
                })
            else:
                # Create simplified detected list for comparison
                simplified_detected = [{
                    'health': d.get('health'),
                    'name': d.get('name'),
                    'is_user': d.get('is_user')
                } for d in detected_players]

                errors.append({
                    'snapshot_id': snapshot_id,
                    'status': 'FAIL',
                    'detected': simplified_detected,
                    'expected': expected_players
                })

        # Print results summary
        print(f"\n{'='*50}")
        print(f"Health Detection Test Results")
        print(f"{'='*50}")
        print(f"Total tests: {len(results) + len(errors)}")
        print(f"Passed: {len(results)}")
        print(f"Failed: {len(errors)}")

        if errors:
            print(f"\nFailed tests (showing first 5):")
            for error in errors[:5]:
                print(f"  Snapshot {error['snapshot_id']:03d}:")
                # Show first mismatch
                for i, (det, exp) in enumerate(zip(error['detected'], error['expected'])):
                    if (det['health'] != exp['health'] or
                        det['name'] != exp['name'] or
                        det['is_user'] != exp['is_user']):
                        print(f"    Player {i}: detected={det} vs expected={exp}")
                        break

        # Calculate accuracy
        total = len(results) + len(errors)
        if total > 0:
            accuracy = (len(results) / total) * 100
            print(f"\nAccuracy: {accuracy:.1f}%")

        # Assert all tests passed
        self.assertEqual(len(errors), 0,
                        f"{len(errors)} tests failed. See details above.")

    def test_health_detection_with_confidence(self):
        """Test health detection with confidence scores"""
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
            detected_players = self.detector.detect_all_players_health(frame)
            expected_players = expected_data['all_players']
            expected_confidence = expected_data['confidence']

            # Check if all players match (comparing only health, name, is_user)
            all_match = True
            for detected, expected in zip(detected_players, expected_players):
                det_health = detected.get('health')
                det_name = detected.get('name')
                det_is_user = detected.get('is_user')

                if (det_health != expected['health'] or
                    det_name != expected['name'] or
                    det_is_user != expected['is_user']):
                    all_match = False
                    break

            results.append({
                'snapshot_id': snapshot_id,
                'expected_confidence': expected_confidence,
                'correct': all_match
            })

        # Analyze confidence correlation with correctness
        correct_results = [r for r in results if r['correct']]
        incorrect_results = [r for r in results if not r['correct']]

        if correct_results:
            avg_confidence_correct = sum(r['expected_confidence'] for r in correct_results) / len(correct_results)
        else:
            avg_confidence_correct = 0

        if incorrect_results:
            avg_confidence_incorrect = sum(r['expected_confidence'] for r in incorrect_results) / len(incorrect_results)
        else:
            avg_confidence_incorrect = 0

        print(f"\n{'='*50}")
        print(f"Confidence Analysis")
        print(f"{'='*50}")
        print(f"Correct detections: {len(correct_results)}/{len(results)}")
        print(f"Average confidence for correct detections: {avg_confidence_correct:.3f}")
        print(f"Average confidence for incorrect detections: {avg_confidence_incorrect:.3f}")

        # Note: We're analyzing ground truth confidence, not detector confidence

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
            detected_result = self.detector.detect_health(frame)

            expected_health = self.ground_truth[snapshot_id]
            detected_health = detected_result.get('health')
            if detected_health is None:
                detected_health = -1

            print(f"\nProblematic case - Snapshot {snapshot_id}:")
            print(f"  Expected: {expected_health}")
            print(f"  Detected: {detected_health}")

            # For problematic cases, we expect exact match
            self.assertEqual(detected_health, expected_health,
                           msg=f"Snapshot {snapshot_id}: Health value mismatch")


def run_tests():
    """Run the test suite"""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()