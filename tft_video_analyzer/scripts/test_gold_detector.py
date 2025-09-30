#!/usr/bin/env python3
"""Test gold detector on sample screenshots"""

import cv2
import os
import glob
from detectors.gold import TFTGoldDetector

def test_gold_detection():
    """Test gold detection on various screenshots"""
    detector = TFTGoldDetector()

    # Get some combat/round screenshots
    screenshots = glob.glob("screenshots/round_*.png")[:20]
    screenshots.extend(glob.glob("screenshots/combat_*.png")[:5])

    print("Testing Gold Detection")
    print("=" * 60)

    detected_count = 0
    results = []

    for filepath in screenshots:
        frame = cv2.imread(filepath)
        if frame is not None:
            gold = detector.detect_gold(frame)
            filename = os.path.basename(filepath)

            if gold is not None:
                detected_count += 1
                results.append((filename, gold))
                print(f"✓ {filename}: {gold} gold")
            else:
                print(f"✗ {filename}: No gold detected")

    print("\n" + "=" * 60)
    print(f"Detection Summary:")
    print(f"  Total screenshots: {len(screenshots)}")
    print(f"  Gold detected: {detected_count}")
    print(f"  Detection rate: {detected_count/len(screenshots)*100:.1f}%")

    if results:
        print(f"\nDetected Gold Amounts:")
        for filename, gold in results[:10]:
            print(f"  {filename}: {gold}g")

    # Save visualization for one screenshot
    if screenshots:
        test_frame = cv2.imread(screenshots[0])
        if test_frame is not None:
            vis_frame = detector.visualize_gold_region(test_frame)
            cv2.imwrite("gold_detection_visualization.png", vis_frame)
            print(f"\nVisualization saved to: gold_detection_visualization.png")

            # Save debug images
            detector.save_debug_images(test_frame, "gold_debug")
            print(f"Debug images saved: gold_debug_*.png")

if __name__ == "__main__":
    test_gold_detection()