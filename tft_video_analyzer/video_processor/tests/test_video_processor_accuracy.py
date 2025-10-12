#!/usr/bin/env python3
"""
Test video processor accuracy against truth data
Compares game states from video processor with truth data from individual detector runs
"""

import json
import pytest
from pathlib import Path
from typing import Dict, List, Tuple


class TestVideoProcessorAccuracy:
    """Test suite for video processor accuracy"""

    @staticmethod
    def load_truth_data(truth_file: Path) -> Dict:
        """Load truth data JSON"""
        with open(truth_file, 'r') as f:
            return json.load(f)

    @staticmethod
    def load_game_states(game_states_file: Path) -> Dict:
        """Load video processor game states JSON"""
        with open(game_states_file, 'r') as f:
            return json.load(f)

    @staticmethod
    def compare_game_states(truth_entry: Dict, processor_state: Dict) -> Tuple[bool, List[str]]:
        """
        Compare a truth entry with a processor state

        Returns:
            Tuple of (matches, differences_list)
        """
        differences = []

        # Compare stage
        if truth_entry.get('stage') != processor_state.get('stage'):
            differences.append(
                f"Stage mismatch: truth={truth_entry.get('stage')} vs processor={processor_state.get('stage')}"
            )

        # Compare round
        if truth_entry.get('round') != processor_state.get('round'):
            differences.append(
                f"Round mismatch: truth={truth_entry.get('round')} vs processor={processor_state.get('round')}"
            )

        # Compare gold
        if truth_entry.get('gold') != processor_state.get('gold'):
            differences.append(
                f"Gold mismatch: truth={truth_entry.get('gold')} vs processor={processor_state.get('gold')}"
            )

        # Compare health (complex object comparison)
        truth_health = truth_entry.get('health')
        processor_health = processor_state.get('health')

        if truth_health != processor_health:
            # For health, we check the actual health value, not the whole dict
            truth_health_val = truth_health.get('health') if isinstance(truth_health, dict) else truth_health
            processor_health_val = processor_health.get('health') if isinstance(processor_health, dict) else processor_health

            if truth_health_val != processor_health_val:
                differences.append(
                    f"Health mismatch: truth={truth_health_val} vs processor={processor_health_val}"
                )

        # Compare streak
        if truth_entry.get('streak') != processor_state.get('streak'):
            differences.append(
                f"Streak mismatch: truth={truth_entry.get('streak')} vs processor={processor_state.get('streak')}"
            )

        return len(differences) == 0, differences

    def test_game_states_match(self, snapshots_dir: str):
        """
        Test that video processor game states match truth data

        Args:
            snapshots_dir: Directory containing snapshots and both JSON files
        """
        snapshots_path = Path(snapshots_dir)

        # Load files
        truth_file = snapshots_path / "truth_game_states.json"
        game_states_file = snapshots_path / "game_states.json"

        assert truth_file.exists(), f"Truth data file not found: {truth_file}"
        assert game_states_file.exists(), f"Game states file not found: {game_states_file}"

        truth_data = self.load_truth_data(truth_file)
        game_states_data = self.load_game_states(game_states_file)

        truth_entries = truth_data['truth_data']
        processor_states = game_states_data['states']

        print(f"\nComparing {len(truth_entries)} truth entries with {len(processor_states)} processor states")

        # Create mappings by snapshot_id for comparison
        truth_by_id = {entry['snapshot_id']: entry for entry in truth_entries}
        processor_by_id = {state['snapshot_id']: state for state in processor_states}

        # Statistics
        total_compared = 0
        total_matches = 0
        total_mismatches = 0
        all_differences = []

        # Compare entries
        for snapshot_id in sorted(truth_by_id.keys()):
            if snapshot_id not in processor_by_id:
                print(f"Warning: Snapshot {snapshot_id} in truth but not in processor output")
                continue

            truth_entry = truth_by_id[snapshot_id]
            processor_state = processor_by_id[snapshot_id]

            matches, differences = self.compare_game_states(truth_entry, processor_state)

            total_compared += 1

            if matches:
                total_matches += 1
            else:
                total_mismatches += 1
                all_differences.append({
                    'snapshot_id': snapshot_id,
                    'filename': truth_entry.get('filename'),
                    'differences': differences
                })

        # Print results
        print(f"\n{'='*70}")
        print("VIDEO PROCESSOR ACCURACY TEST RESULTS")
        print(f"{'='*70}")
        print(f"Total snapshots compared: {total_compared}")
        print(f"Exact matches: {total_matches} ({total_matches/total_compared*100:.1f}%)")
        print(f"Mismatches: {total_mismatches} ({total_mismatches/total_compared*100:.1f}%)")

        if all_differences:
            print(f"\n{'='*70}")
            print("MISMATCHES DETAILS")
            print(f"{'='*70}")
            for i, diff_info in enumerate(all_differences[:10], 1):  # Show first 10
                print(f"\n{i}. Snapshot {diff_info['snapshot_id']} ({diff_info['filename']})")
                for diff in diff_info['differences']:
                    print(f"   - {diff}")

            if len(all_differences) > 10:
                print(f"\n... and {len(all_differences) - 10} more mismatches")

        # Assert high accuracy (allow some tolerance for OCR variability)
        accuracy = total_matches / total_compared if total_compared > 0 else 0
        assert accuracy >= 0.90, f"Accuracy too low: {accuracy*100:.1f}% (expected >= 90%)"

        print(f"\n✓ Test passed with {accuracy*100:.1f}% accuracy")

    def test_critical_fields_match(self, snapshots_dir: str):
        """
        Test that critical fields (stage, round) have 100% accuracy

        Args:
            snapshots_dir: Directory containing snapshots and both JSON files
        """
        snapshots_path = Path(snapshots_dir)

        truth_file = snapshots_path / "truth_game_states.json"
        game_states_file = snapshots_path / "game_states.json"

        truth_data = self.load_truth_data(truth_file)
        game_states_data = self.load_game_states(game_states_file)

        truth_entries = truth_data['truth_data']
        processor_states = game_states_data['states']

        # Create mappings
        truth_by_id = {entry['snapshot_id']: entry for entry in truth_entries}
        processor_by_id = {state['snapshot_id']: state for state in processor_states}

        stage_mismatches = 0
        round_mismatches = 0
        total = 0

        for snapshot_id in sorted(truth_by_id.keys()):
            if snapshot_id not in processor_by_id:
                continue

            truth_entry = truth_by_id[snapshot_id]
            processor_state = processor_by_id[snapshot_id]

            total += 1

            if truth_entry.get('stage') != processor_state.get('stage'):
                stage_mismatches += 1

            if truth_entry.get('round') != processor_state.get('round'):
                round_mismatches += 1

        print(f"\n{'='*70}")
        print("CRITICAL FIELDS ACCURACY")
        print(f"{'='*70}")
        print(f"Stage accuracy: {(total-stage_mismatches)/total*100:.1f}% ({total-stage_mismatches}/{total})")
        print(f"Round accuracy: {(total-round_mismatches)/total*100:.1f}% ({total-round_mismatches}/{total})")

        # Critical fields should be 100% accurate
        assert stage_mismatches == 0, f"Stage detection has {stage_mismatches} mismatches"
        assert round_mismatches == 0, f"Round detection has {round_mismatches} mismatches"

        print("\n✓ Critical fields test passed")


def run_accuracy_test(snapshots_dir: str):
    """
    Run accuracy tests on a snapshots directory

    Args:
        snapshots_dir: Directory containing snapshots and both JSON files
    """
    tester = TestVideoProcessorAccuracy()

    print(f"Running accuracy tests on: {snapshots_dir}\n")

    try:
        tester.test_critical_fields_match(snapshots_dir)
        tester.test_game_states_match(snapshots_dir)
        print("\n" + "="*70)
        print("ALL TESTS PASSED")
        print("="*70)
    except AssertionError as e:
        print("\n" + "="*70)
        print(f"TEST FAILED: {e}")
        print("="*70)
        raise


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_video_processor_accuracy.py <snapshots_directory>")
        sys.exit(1)

    snapshots_dir = sys.argv[1]
    run_accuracy_test(snapshots_dir)
