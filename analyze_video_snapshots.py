#!/usr/bin/env python3
"""Extract random frames from TFT video and run detectors on each frame"""

import sys
import os
import cv2
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import detectors
from tft_video_analyzer.detectors.streak.detector import TFTStreakDetector
from tft_video_analyzer.detectors.gold.detector import TFTGoldDetector
from tft_video_analyzer.detectors.health.detector import TFTHealthDetector

def convert_numpy_types(obj):
    """Convert numpy types to Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def extract_random_frames(video_path, num_frames=33):
    """Extract random frames from video"""
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f} seconds")

    # Generate random frame numbers (avoid very beginning and end)
    margin = int(fps * 5)  # Skip first and last 5 seconds
    frame_numbers = sorted(random.sample(
        range(margin, total_frames - margin),
        min(num_frames, total_frames - 2 * margin)
    ))

    frames_data = []

    for idx, frame_num in enumerate(frame_numbers):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if ret:
            timestamp = frame_num / fps
            frames_data.append({
                'index': idx,
                'frame_number': frame_num,
                'timestamp': timestamp,
                'frame': frame
            })
            print(f"Extracted frame {idx+1}/{num_frames} at {timestamp:.2f}s")

    cap.release()
    return frames_data

def run_detectors(frame, snapshot_dir=None):
    """Run all detectors on a single frame and optionally save regions"""
    results = {}

    # Initialize detectors
    streak_detector = TFTStreakDetector()
    gold_detector = TFTGoldDetector()
    health_detector = TFTHealthDetector()

    # Save detection regions if snapshot_dir is provided
    if snapshot_dir:
        # Extract and save streak region
        streak_region = streak_detector.extract_streak_region(frame)
        streak_region_path = snapshot_dir / "streak_region.png"
        cv2.imwrite(str(streak_region_path), streak_region)

        # Extract and save gold region
        gold_region = gold_detector.extract_gold_region(frame)
        gold_region_path = snapshot_dir / "gold_region.png"
        cv2.imwrite(str(gold_region_path), gold_region)

        # Save preprocessed versions for debugging
        try:
            # Preprocessed streak region
            streak_processed = streak_detector.preprocess_for_ocr(streak_region)
            streak_processed_path = snapshot_dir / "streak_region_processed.png"
            cv2.imwrite(str(streak_processed_path), streak_processed)
        except:
            pass

        try:
            # Preprocessed gold region
            gold_processed = gold_detector.preprocess_for_ocr(gold_region)
            gold_processed_path = snapshot_dir / "gold_region_processed.png"
            cv2.imwrite(str(gold_processed_path), gold_processed)
        except:
            pass

        # Save visualization images
        streak_vis = streak_detector.visualize_streak_region(frame)
        streak_vis_path = snapshot_dir / "streak_visualization.png"
        cv2.imwrite(str(streak_vis_path), streak_vis)

        gold_vis = gold_detector.visualize_gold_region(frame)
        gold_vis_path = snapshot_dir / "gold_visualization.png"
        cv2.imwrite(str(gold_vis_path), gold_vis)

    # Run streak detector
    try:
        streak_data = streak_detector.detect_streak(frame)
        # Convert StreakType.NONE to empty string as requested
        streak_type = streak_data['type'].value if streak_data['type'] else 'none'
        if streak_type == 'none':
            streak_type = ''  # Empty string for no streak detected

        results['streak'] = {
            'type': streak_type,
            'length': streak_data['length'] if streak_data['length'] is not None else -1,
            'confidence': streak_data.get('confidence', 0.0)
        }
    except Exception as e:
        results['streak'] = {'error': str(e)}

    # Run gold detector with confidence
    try:
        gold_amount, gold_confidence = gold_detector.detect_gold_with_confidence(frame)
        results['gold'] = {
            'amount': gold_amount if gold_amount is not None else -1,
            'confidence': gold_confidence
        }
    except Exception as e:
        results['gold'] = {'error': str(e)}

    # Run health detector
    try:
        # Use detect_health for user's health
        health_data = health_detector.detect_health(frame)
        results['health'] = {
            'user_health': health_data.get('health', None),
            'found': health_data.get('found', False),
            'confidence': health_data.get('confidence', 0.0)
        }

        # Also try to detect all players
        all_players = health_detector.detect_all_players_health(frame)
        if all_players:
            results['health']['all_players'] = all_players
    except Exception as e:
        results['health'] = {'error': str(e)}

    return results

def main():
    # Video path
    video_path = "tft_video_analyzer/videos/League of Legends_09-20-2025_15-28-29-883.mp4"

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"video_analysis_{timestamp}")
    output_dir.mkdir(exist_ok=True)

    print(f"Created output directory: {output_dir}")

    # Extract random frames
    print("\nExtracting 33 random frames from video...")
    frames_data = extract_random_frames(video_path, 33)

    # Process each frame
    all_results = []

    for frame_data in frames_data:
        idx = frame_data['index']
        frame = frame_data['frame']

        # Create directory for this snapshot
        snapshot_dir = output_dir / f"snapshot_{idx:03d}"
        snapshot_dir.mkdir(exist_ok=True)

        # Save the frame
        frame_path = snapshot_dir / "frame.png"
        cv2.imwrite(str(frame_path), frame)

        print(f"\nProcessing snapshot {idx+1}/33...")
        print(f"  Frame: {frame_data['frame_number']}, Time: {frame_data['timestamp']:.2f}s")

        # Run detectors and save regions
        detection_results = run_detectors(frame, snapshot_dir)

        # Add metadata
        result_entry = {
            'snapshot_id': idx,
            'frame_number': frame_data['frame_number'],
            'timestamp_seconds': frame_data['timestamp'],
            'frame_path': str(frame_path),
            'detections': detection_results
        }

        # Convert numpy types for JSON serialization
        result_entry = convert_numpy_types(result_entry)

        # Save individual result
        result_path = snapshot_dir / "detection_results.json"
        with open(result_path, 'w') as f:
            json.dump(result_entry, f, indent=2)

        all_results.append(result_entry)

        # Print detection summary
        if 'streak' in detection_results and 'error' not in detection_results['streak']:
            streak_type = detection_results['streak']['type']
            streak_length = detection_results['streak']['length']
            streak_conf = detection_results['streak']['confidence']

            if streak_type == '' and streak_length == -1:
                print(f"  Streak: Not detected (conf: {streak_conf:.2f})")
            else:
                print(f"  Streak: {streak_type if streak_type else 'unknown'} (length: {streak_length}, conf: {streak_conf:.2f})")
        if 'gold' in detection_results and 'error' not in detection_results['gold']:
            gold_amount = detection_results['gold']['amount']
            gold_conf = detection_results['gold'].get('confidence', 0.0)
            if gold_amount != -1:
                print(f"  Gold: {gold_amount} (conf: {gold_conf:.2f})")
            else:
                print(f"  Gold: Not detected (conf: {gold_conf:.2f})")
        if 'health' in detection_results and 'error' not in detection_results['health']:
            if detection_results['health'].get('found'):
                print(f"  Health: {detection_results['health']['user_health']} (conf: {detection_results['health']['confidence']:.2f})")
            if 'all_players' in detection_results['health']:
                print(f"  All players: {len(detection_results['health']['all_players'])} detected")

    # Save combined results
    combined_path = output_dir / "combined_results.json"
    with open(combined_path, 'w') as f:
        json.dump({
            'video_path': video_path,
            'analysis_timestamp': timestamp,
            'total_snapshots': len(all_results),
            'snapshots': all_results
        }, f, indent=2)

    print(f"\n✅ Analysis complete!")
    print(f"Output directory: {output_dir}")
    print(f"Combined results: {combined_path}")
    print(f"Total snapshots: {len(all_results)}")

if __name__ == "__main__":
    main()