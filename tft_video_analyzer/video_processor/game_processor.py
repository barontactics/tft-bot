#!/usr/bin/env python3
"""
TFT Game Video Processor
Processes TFT gameplay videos with adaptive sampling based on game state detection
"""

import cv2
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import time

# Import detectors
from ..detectors.round.detector import TFTRoundDetector
from ..detectors.gold.detector import TFTGoldDetector
from ..detectors.health.detector import TFTHealthDetector
from ..detectors.streak.detector import TFTStreakDetector
from ..models.game_state import GameState


class TFTGameProcessor:
    """Process TFT videos with adaptive frame sampling"""

    def __init__(self, video_path: str, output_dir: str = "snapshots"):
        """
        Initialize video processor

        Args:
            video_path: Path to the video file
            output_dir: Directory to save snapshots
        """
        self.video_path = video_path
        self.output_dir = output_dir

        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

        # Initialize detectors
        self.round_detector = TFTRoundDetector()
        self.gold_detector = TFTGoldDetector()
        self.health_detector = TFTHealthDetector()
        self.streak_detector = TFTStreakDetector()

        # Game state tracking
        self.current_game_state = GameState()
        self.game_started = False  # Set to True once we detect stage-round
        self.detectors_enabled = False  # Enable gold/health/streak after round 1-2
        self.game_state_snapshots: List[GameState] = []  # Store all game states

        # Frame tracking
        self.frame_count = 0
        self.snapshot_count = 0

        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(video_path).stem
        self.session_dir = Path(output_dir) / f"{video_name}_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Metadata tracking
        self.snapshots_metadata = []

        print(f"Video Processor initialized")
        print(f"  Video: {Path(video_path).name}")
        print(f"  Duration: {self.duration:.1f} seconds")
        print(f"  FPS: {self.fps:.1f}")
        print(f"  Total frames: {self.total_frames}")
        print(f"  Output: {self.session_dir}")

    def detect_round(self, frame) -> Optional[tuple]:
        """Detect stage and round from frame"""
        return self.round_detector.detect_stage_round(frame)

    def save_snapshot(self, frame, stage: Optional[int] = None, round_num: Optional[int] = None):
        """Save a snapshot with metadata"""

        # Determine filename
        timestamp_sec = self.frame_count / self.fps
        self.snapshot_count += 1

        if stage and round_num:
            filename = f"snapshot_{self.snapshot_count:04d}_stage{stage}-{round_num}_frame{self.frame_count}_{timestamp_sec:.1f}s.png"
        else:
            filename = f"snapshot_{self.snapshot_count:04d}_loading_frame{self.frame_count}_{timestamp_sec:.1f}s.png"

        filepath = self.session_dir / filename

        # Save image
        cv2.imwrite(str(filepath), frame)

        # Save metadata
        metadata = {
            'snapshot_id': self.snapshot_count,
            'filename': filename,
            'frame_number': self.frame_count,
            'timestamp': timestamp_sec,
            'stage': stage,
            'round': round_num,
            'game_started': self.game_started
        }

        self.snapshots_metadata.append(metadata)

        # Print status
        if stage and round_num:
            print(f"  📸 Snapshot {self.snapshot_count:04d}: Stage {stage}-{round_num} @ {timestamp_sec:.1f}s")
        else:
            print(f"  📸 Snapshot {self.snapshot_count:04d}: Loading @ {timestamp_sec:.1f}s")

    def process(self):
        """
        Process the video with adaptive sampling:
        - Sample every 2 seconds initially (looking for game start)
        - Once stage-round detected, switch to 1 second sampling
        """
        print(f"\nProcessing video...")
        print(f"  Initial sampling: Every 2 seconds")
        print(f"  After game start: Every 1 second")
        print("-" * 60)

        # Sampling configuration
        initial_sample_interval = 2.0  # seconds
        game_sample_interval = 1.0     # seconds

        # Calculate frame intervals
        initial_frame_interval = int(self.fps * initial_sample_interval)
        game_frame_interval = int(self.fps * game_sample_interval)

        # Current sampling mode
        current_frame_interval = initial_frame_interval

        # Progress tracking
        start_time = time.time()
        last_progress_update = 0

        while True:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process frame at current interval
            if self.frame_count % current_frame_interval == 0:

                # Detect stage and round
                stage_round = self.detect_round(frame)

                if stage_round:
                    stage = stage_round['stage']
                    round_num = stage_round['round']

                    # Check if game just started
                    if not self.game_started:
                        self.game_started = True
                        current_frame_interval = game_frame_interval
                        print(f"\n🎮 Game started! Detected Stage {stage}-{round_num}")
                        print(f"  Switching to {game_sample_interval}s sampling\n")

                    # Enable other detectors once we reach stage 1, round 2
                    if not self.detectors_enabled and stage >= 1 and round_num >= 2:
                        self.detectors_enabled = True
                        print(f"\n🔍 Enabling gold, health, and streak detectors at Stage {stage}-{round_num}\n")

                    # Update game state with stage and round
                    self.current_game_state.update(stage=stage, round=round_num, frame_number=self.frame_count)

                    # Run additional detectors if enabled
                    if self.detectors_enabled:
                        # Detect gold
                        gold = self.gold_detector.detect_gold(frame)
                        if gold is not None:
                            self.current_game_state.update(gold=gold)

                        # Detect health
                        health = self.health_detector.detect_health(frame)
                        if health is not None:
                            self.current_game_state.update(health=health)

                        # Detect streak
                        streak_result = self.streak_detector.detect_streak(frame)
                        if streak_result:
                            # streak_result is a dict with 'type' and 'length' keys
                            streak_type = streak_result.get('type')
                            streak_length = streak_result.get('length')

                            if streak_length and streak_length > 0:
                                # Convert to signed value: positive for win, negative for loss
                                if streak_type.value == 'loss':
                                    streak_value = -streak_length
                                else:
                                    streak_value = streak_length
                                self.current_game_state.update(streak=streak_value)

                    # Save a snapshot of the current game state (before saving snapshot image)
                    # Note: snapshot_count will be incremented in save_snapshot()
                    snapshot_id = self.snapshot_count + 1

                    state_snapshot = GameState(
                        stage=self.current_game_state.stage,
                        round=self.current_game_state.round,
                        gold=self.current_game_state.gold,
                        health=self.current_game_state.health,
                        streak=self.current_game_state.streak,
                        frame_number=self.frame_count,
                        timestamp=datetime.now()
                    )
                    self.game_state_snapshots.append((snapshot_id, state_snapshot))

                    # Print current state
                    print(f"  {state_snapshot}")

                    # Save snapshot image
                    self.save_snapshot(frame, stage, round_num)

                else:
                    # No stage-round detected yet (still in loading/pre-game)
                    if not self.game_started:
                        # Save snapshots during loading phase
                        self.save_snapshot(frame)

                # Progress update every 5 seconds
                current_time = time.time()
                if current_time - last_progress_update >= 5:
                    progress = (self.frame_count / self.total_frames) * 100
                    elapsed = current_time - start_time
                    fps_processed = self.frame_count / elapsed if elapsed > 0 else 0
                    eta = (self.total_frames - self.frame_count) / (fps_processed * self.fps) if fps_processed > 0 else 0

                    print(f"  Progress: {progress:.1f}% | Snapshots: {self.snapshot_count} | ETA: {eta:.0f}s")
                    last_progress_update = current_time

            self.frame_count += 1

        # Cleanup
        self.cap.release()

        # Save metadata and game states
        self.save_session_metadata()
        self.save_game_states()

        # Print summary
        self.print_summary()

    def save_session_metadata(self):
        """Save session metadata to JSON file"""
        metadata = {
            'video_path': self.video_path,
            'duration': self.duration,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'frames_processed': self.frame_count,
            'snapshots_saved': self.snapshot_count,
            'game_started': self.game_started,
            'detectors_enabled': self.detectors_enabled,
            'snapshots': self.snapshots_metadata,
            'session_timestamp': datetime.now().isoformat()
        }

        metadata_path = self.session_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n  Metadata saved: {metadata_path}")

    def save_game_states(self):
        """Save game state snapshots to JSON file"""
        # Convert (snapshot_id, state) tuples to dicts with snapshot_id and snapshot metadata
        states_with_ids = []
        for snapshot_id, state in self.game_state_snapshots:
            state_dict = state.to_dict()
            state_dict['snapshot_id'] = snapshot_id

            # Find matching snapshot metadata
            snapshot_meta = None
            for meta in self.snapshots_metadata:
                if meta['snapshot_id'] == snapshot_id:
                    snapshot_meta = {
                        'filename': meta['filename'],
                        'timestamp': meta['timestamp']
                    }
                    break

            state_dict['snapshot'] = snapshot_meta
            states_with_ids.append(state_dict)

        game_states_data = {
            'total_states': len(self.game_state_snapshots),
            'detectors_enabled': self.detectors_enabled,
            'states': states_with_ids
        }

        game_states_path = self.session_dir / 'game_states.json'
        with open(game_states_path, 'w') as f:
            json.dump(game_states_data, f, indent=2)

        print(f"  Game states saved: {game_states_path}")

    def print_summary(self):
        """Print processing summary"""
        print("\n" + "=" * 60)
        print("Processing Complete!")
        print(f"  Total snapshots: {self.snapshot_count}")
        print(f"  Game started: {'Yes' if self.game_started else 'No'}")
        print(f"  Detectors enabled: {'Yes' if self.detectors_enabled else 'No'}")
        print(f"  Game states captured: {len(self.game_state_snapshots)}")

        if self.game_started:
            # Count unique rounds
            rounds_seen = set()
            for snapshot in self.snapshots_metadata:
                if snapshot['stage'] and snapshot['round']:
                    rounds_seen.add((snapshot['stage'], snapshot['round']))

            print(f"  Unique rounds detected: {len(rounds_seen)}")
            if rounds_seen:
                print(f"  Rounds: {', '.join(f'{s}-{r}' for s, r in sorted(rounds_seen))}")

        if self.game_state_snapshots:
            # Summary of game states
            complete_states = sum(1 for _, state in self.game_state_snapshots if state.is_complete())
            print(f"\n  Complete game states: {complete_states}/{len(self.game_state_snapshots)}")

            # Show first and last state
            if len(self.game_state_snapshots) > 0:
                print(f"\n  First state: {self.game_state_snapshots[0][1]}")
                if len(self.game_state_snapshots) > 1:
                    print(f"  Last state:  {self.game_state_snapshots[-1][1]}")

        print(f"\n  Output directory: {self.session_dir}")
        print("=" * 60)


def main():
    """Main function for testing"""
    import argparse

    parser = argparse.ArgumentParser(description="TFT Game Video Processor")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("-o", "--output", default="snapshots", help="Output directory")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return

    try:
        processor = TFTGameProcessor(args.video, args.output)
        processor.process()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
