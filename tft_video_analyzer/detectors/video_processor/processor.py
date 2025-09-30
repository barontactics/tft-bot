#!/usr/bin/env python3
"""
TFT Video Processor
Processes TFT gameplay videos, detecting game states and capturing key moments
"""

import cv2
import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import time

# Import all our detectors
from ..loading import TFTLoadingScreenDetector
from ..planning import TFTPlanningDetector
from ..augment import TFTAugmentDetector
from ..game_state import TFTGameState, GamePhase

class TFTVideoProcessor:
    """Process TFT videos and capture key moments"""

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
        self.loading_detector = TFTLoadingScreenDetector()
        self.planning_detector = TFTPlanningDetector()
        self.augment_detector = TFTAugmentDetector()

        # Game state manager
        self.game_state = TFTGameState()

        # Tracking variables
        self.snapshots_saved = []
        self.loading_screen_seen = False
        self.game_started = False
        self.last_detection_state = {}
        self.frame_count = 0

        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.session_dir = os.path.join(output_dir, f"{video_name}_{timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)

        # Create special directories for loading and unknown
        self.special_dirs = {
            'loading': os.path.join(self.session_dir, 'loading'),
            'unknown': os.path.join(self.session_dir, 'unknown')
        }

        for dir_path in self.special_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        print(f"Video Processor initialized")
        print(f"  Video: {os.path.basename(video_path)}")
        print(f"  Duration: {self.duration:.1f} seconds")
        print(f"  FPS: {self.fps:.1f}")
        print(f"  Output: {self.session_dir}")

    def detect_current_state(self, frame) -> Dict:
        """Run all detectors on current frame"""
        detections = {
            'is_loading': self.loading_detector.detect_loading_screen(frame),
            'is_planning_start': self.planning_detector.detect_planning_phase(frame),
            'is_augment': self.augment_detector.detect_augment_selection(frame),
            'timestamp': self.frame_count / self.fps
        }
        return detections

    def update_game_state(self, detections: Dict):
        """Update game state based on detections"""

        # Check if game has started (after loading screen)
        if detections['is_loading']:
            self.loading_screen_seen = True
            print(f"  [{detections['timestamp']:.1f}s] Loading screen detected")

        # Start game after loading screen disappears
        if self.loading_screen_seen and not self.game_started and not detections['is_loading']:
            self.game_started = True
            self.game_state.start_game()
            print(f"  [{detections['timestamp']:.1f}s] Game started")

        # Update phases if game has started
        if self.game_started:
            # Detect augment phase
            if detections['is_augment']:
                if self.game_state.phase != GamePhase.AUGMENT:
                    self.game_state.set_phase(GamePhase.AUGMENT)
                    print(f"  [{detections['timestamp']:.1f}s] Augment selection detected")

            # Detect planning phase start
            elif detections['is_planning_start']:
                if self.game_state.phase != GamePhase.PLANNING:
                    self.game_state.set_phase(GamePhase.PLANNING)
                    print(f"  [{detections['timestamp']:.1f}s] Planning phase started")

                    # If it's a new planning phase, might be a new round
                    # In real implementation, would detect round number here
                    if self.game_state.phase_start_time is None:
                        self.game_state.phase_start_time = detections['timestamp']

            # Detect transition to combat (when planning text disappears)
            elif self.last_detection_state.get('is_planning_start', False) and not detections['is_planning_start']:
                if self.game_state.phase == GamePhase.PLANNING:
                    self.game_state.transition_to_combat()
                    print(f"  [{detections['timestamp']:.1f}s] Combat phase started")

    def should_save_snapshot(self, detections: Dict) -> bool:
        """Determine if current frame should be saved"""

        if not self.game_started:
            # Save loading screen once
            if detections['is_loading'] and not any(s['type'] == 'loading' for s in self.snapshots_saved):
                return True
            return False

        # Save key moments
        if detections['is_augment']:
            # Save augment selection if we haven't saved one recently
            recent_augment = any(
                s['type'] == 'augment' and
                abs(s['timestamp'] - detections['timestamp']) < 5
                for s in self.snapshots_saved
            )
            if not recent_augment:
                return True

        if detections['is_planning_start']:
            # Save planning phase start
            recent_planning = any(
                s['type'] == 'planning' and
                abs(s['timestamp'] - detections['timestamp']) < 10
                for s in self.snapshots_saved
            )
            if not recent_planning:
                return True

        # Save combat start (right after planning ends)
        if self.game_state.phase == GamePhase.COMBAT and self.game_state.phase_start_time == detections['timestamp']:
            return True

        return False

    def save_snapshot(self, frame, detections: Dict, snapshot_type: str):
        """Save a snapshot with metadata"""

        # Create filename based on detection type
        timestamp_str = f"{detections['timestamp']:.1f}s"
        stage = self.game_state.stage
        round_num = self.game_state.round

        # Count existing snapshots of this type for numbering
        existing_count = len([s for s in self.snapshots_saved if s['type'] == snapshot_type])
        counter = existing_count + 1

        # Determine directory based on snapshot type
        if snapshot_type == 'loading' or (stage == 0 and round_num == 0):
            # Loading screens go in special loading directory
            snapshot_dir = self.special_dirs['loading']
            filename = f"loading_{counter:03d}_{timestamp_str}.png"
        elif stage > 0 and round_num > 0:
            # Game snapshots go in stage-round directories
            stage_round_dir = os.path.join(self.session_dir, f"stage{stage}-{round_num}")

            # Create stage-round directory if it doesn't exist
            if not os.path.exists(stage_round_dir):
                os.makedirs(stage_round_dir)

            snapshot_dir = stage_round_dir
            filename = f"{snapshot_type}_{counter:03d}_{timestamp_str}.png"
        else:
            # Unknown state snapshots
            snapshot_dir = self.special_dirs['unknown']
            filename = f"{snapshot_type}_{counter:03d}_{timestamp_str}.png"

        filepath = os.path.join(snapshot_dir, filename)

        # Save image
        cv2.imwrite(filepath, frame)

        # Save metadata
        metadata = {
            'filename': filename,
            'type': snapshot_type,
            'timestamp': detections['timestamp'],
            'stage': self.game_state.stage,
            'round': self.game_state.round,
            'phase': self.game_state.phase.value,
            'detections': detections,
            'game_state': self.game_state.get_state_dict()
        }

        self.snapshots_saved.append(metadata)

        # Print save location with stage-round info
        if stage > 0 and round_num > 0:
            print(f"  📸 Saved: stage{stage}-{round_num}/{filename}")
        else:
            print(f"  📸 Saved: {os.path.basename(os.path.dirname(filepath))}/{filename}")

        return filepath

    def process(self, sample_rate: float = 1.0, max_snapshots: Optional[int] = None):
        """
        Process the video

        Args:
            sample_rate: Sample every N seconds (default: 1.0)
            max_snapshots: Maximum number of snapshots to save (None for unlimited)
        """
        print(f"\nProcessing video (sampling every {sample_rate}s)...")
        print("-" * 60)

        # Calculate frame interval
        frame_interval = int(self.fps * sample_rate)

        # Progress tracking
        start_time = time.time()
        frames_processed = 0

        while True:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process frame at interval
            if self.frame_count % frame_interval == 0:
                frames_processed += 1

                # Run detections
                detections = self.detect_current_state(frame)

                # Update game state
                self.update_game_state(detections)

                # Determine if we should save snapshot
                if self.should_save_snapshot(detections):
                    # Determine snapshot type
                    if detections['is_loading']:
                        snapshot_type = 'loading'
                    elif detections['is_augment']:
                        snapshot_type = 'augment'
                    elif detections['is_planning_start']:
                        snapshot_type = 'planning'
                    elif self.game_state.phase == GamePhase.COMBAT:
                        snapshot_type = 'combat'
                    else:
                        snapshot_type = 'unknown'

                    # Save snapshot
                    self.save_snapshot(frame, detections, snapshot_type)

                    # Check max snapshots
                    if max_snapshots and len(self.snapshots_saved) >= max_snapshots:
                        print(f"\nReached maximum snapshots ({max_snapshots})")
                        break

                # Update last state
                self.last_detection_state = detections

                # Progress update every 10 seconds
                if frames_processed % 10 == 0:
                    progress = (self.frame_count / self.total_frames) * 100
                    elapsed = time.time() - start_time
                    eta = (elapsed / (self.frame_count + 1)) * (self.total_frames - self.frame_count)
                    print(f"  Progress: {progress:.1f}% | ETA: {eta:.0f}s")

            self.frame_count += 1

        # Cleanup
        self.cap.release()

        # Save metadata
        self.save_session_metadata()

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
            'snapshots_saved': len(self.snapshots_saved),
            'snapshots': self.snapshots_saved,
            'session_timestamp': datetime.now().isoformat()
        }

        metadata_path = os.path.join(self.session_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def print_summary(self):
        """Print processing summary"""
        print("\n" + "=" * 60)
        print("Processing Complete!")
        print(f"  Total snapshots: {len(self.snapshots_saved)}")

        # Count by type
        type_counts = {}
        for snapshot in self.snapshots_saved:
            snap_type = snapshot['type']
            type_counts[snap_type] = type_counts.get(snap_type, 0) + 1

        print("\n  Snapshots by type:")
        for snap_type, count in sorted(type_counts.items()):
            print(f"    - {snap_type}: {count}")

        print(f"\n  Output directory: {self.session_dir}")


def main():
    """Main function for testing"""
    import argparse

    parser = argparse.ArgumentParser(description="TFT Video Processor")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("-o", "--output", default="snapshots", help="Output directory")
    parser.add_argument("-r", "--rate", type=float, default=1.0, help="Sample rate in seconds")
    parser.add_argument("-m", "--max", type=int, help="Maximum snapshots to save")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return

    try:
        processor = TFTVideoProcessor(args.video, args.output)
        processor.process(sample_rate=args.rate, max_snapshots=args.max)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()