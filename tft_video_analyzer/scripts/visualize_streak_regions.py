#!/usr/bin/env python3
"""Visualize streak detection regions and processing steps"""

import cv2
import numpy as np
import os
from detectors.streak import TFTStreakDetector, StreakType
from detectors.gold import TFTGoldDetector

def create_streak_visualizations():
    """Create comprehensive visualizations of streak detection"""

    detector = TFTStreakDetector()
    gold_detector = TFTGoldDetector()  # For reference
    output_dir = "detectors/streak/streak_region"

    # Test on multiple screenshots
    test_screenshots = [
        "screenshots/round_1-3_frame03960_time01m06s.png",
        "screenshots/round_2-3_frame18720_time05m12s.png",
        "screenshots/round_2-2_frame16200_time04m30s.png",
        "screenshots/round_3-5_frame53820_time14m57s.png",
    ]

    # Find first valid screenshot
    test_frame = None
    test_name = None
    for screenshot in test_screenshots:
        if os.path.exists(screenshot):
            test_frame = cv2.imread(screenshot)
            test_name = os.path.basename(screenshot).replace('.png', '')
            break

    if test_frame is None:
        print("No test screenshots found")
        return

    h, w = test_frame.shape[:2]
    print(f"Processing {test_name}")
    print(f"Image dimensions: {w}x{h}")

    # 1. Full frame with streak and gold regions highlighted
    vis_full = test_frame.copy()

    # Draw gold region
    gold_x1 = int(w * gold_detector.gold_region['x1_rel'])
    gold_x2 = int(w * gold_detector.gold_region['x2_rel'])
    gold_y1 = int(h * gold_detector.gold_region['y1_rel'])
    gold_y2 = int(h * gold_detector.gold_region['y2_rel'])
    cv2.rectangle(vis_full, (gold_x1, gold_y1), (gold_x2, gold_y2), (255, 215, 0), 2)
    cv2.putText(vis_full, "Gold", (gold_x1, gold_y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 215, 0), 2)

    # Draw streak region
    x1 = int(w * detector.streak_region['x1_rel'])
    x2 = int(w * detector.streak_region['x2_rel'])
    y1 = int(h * detector.streak_region['y1_rel'])
    y2 = int(h * detector.streak_region['y2_rel'])

    # Detect streak for color
    result = detector.detect_streak(test_frame)
    if result['type'] == StreakType.WIN:
        color = (0, 0, 255)  # Red
        label = "Win Streak"
    elif result['type'] == StreakType.LOSS:
        color = (255, 0, 0)  # Blue
        label = "Loss Streak"
    else:
        color = (128, 128, 128)  # Gray
        label = "No Streak"

    cv2.rectangle(vis_full, (x1, y1), (x2, y2), color, 3)
    cv2.putText(vis_full, label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Add coordinate info
    coord_text = f"({detector.streak_region['x1_rel']:.1%}, {detector.streak_region['y1_rel']:.1%}) to ({detector.streak_region['x2_rel']:.1%}, {detector.streak_region['y2_rel']:.1%})"
    cv2.putText(vis_full, coord_text, (x1, y2+30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(os.path.join(output_dir, "1_full_frame_with_regions.png"), vis_full)

    # 2. Extracted streak region
    streak_region = detector.extract_streak_region(test_frame)
    cv2.imwrite(os.path.join(output_dir, "2_extracted_region.png"), streak_region)

    # 3. Color masks
    hsv = cv2.cvtColor(streak_region, cv2.COLOR_BGR2HSV)

    # Blue mask (loss streak)
    blue_mask = cv2.inRange(hsv, detector.blue_lower, detector.blue_upper)
    cv2.imwrite(os.path.join(output_dir, "3_blue_mask_loss.png"), blue_mask)

    # Red mask (win streak)
    red_mask1 = cv2.inRange(hsv, detector.red_lower1, detector.red_upper1)
    red_mask2 = cv2.inRange(hsv, detector.red_lower2, detector.red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    cv2.imwrite(os.path.join(output_dir, "4_red_mask_win.png"), red_mask)

    # 5. Preprocessed for OCR
    processed = detector.preprocess_for_ocr(streak_region)
    cv2.imwrite(os.path.join(output_dir, "5_preprocessed_for_ocr.png"), processed)

    # 6. Process multiple screenshots and create a grid
    grid_images = []
    for i, screenshot_path in enumerate(test_screenshots[:4]):
        if os.path.exists(screenshot_path):
            frame = cv2.imread(screenshot_path)
            if frame is not None:
                # Draw both regions
                vis = frame.copy()
                h, w = frame.shape[:2]

                # Gold region
                gold_x1 = int(w * gold_detector.gold_region['x1_rel'])
                gold_x2 = int(w * gold_detector.gold_region['x2_rel'])
                gold_y1 = int(h * gold_detector.gold_region['y1_rel'])
                gold_y2 = int(h * gold_detector.gold_region['y2_rel'])
                cv2.rectangle(vis, (gold_x1, gold_y1), (gold_x2, gold_y2), (255, 215, 0), 2)

                # Streak region
                x1 = int(w * detector.streak_region['x1_rel'])
                x2 = int(w * detector.streak_region['x2_rel'])
                y1 = int(h * detector.streak_region['y1_rel'])
                y2 = int(h * detector.streak_region['y2_rel'])

                # Detect streak
                result = detector.detect_streak(frame)
                if result['type'] == StreakType.WIN:
                    color = (0, 0, 255)
                elif result['type'] == StreakType.LOSS:
                    color = (255, 0, 0)
                else:
                    color = (128, 128, 128)

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)

                # Add labels
                streak_text = f"{result['type'].value}: {result['length'] if result['length'] is not None else '?'}"
                cv2.putText(vis, streak_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Resize for grid
                vis_small = cv2.resize(vis, (960, 540))
                grid_images.append(vis_small)

                # Save individual visualization
                name = os.path.basename(screenshot_path).replace('.png', '')
                cv2.imwrite(os.path.join(output_dir, f"sample_{i+1}_{name}.png"), vis)

    # 7. Create comparison grid if we have multiple images
    if len(grid_images) >= 2:
        # Create 2x2 grid
        while len(grid_images) < 4:
            grid_images.append(np.zeros_like(grid_images[0]))

        row1 = np.hstack([grid_images[0], grid_images[1]])
        row2 = np.hstack([grid_images[2], grid_images[3]])
        grid = np.vstack([row1, row2])

        cv2.imwrite(os.path.join(output_dir, "6_multiple_samples_grid.png"), grid)

    # 8. Processing stages comparison
    stages_vis = []

    # Original region
    region_resized = cv2.resize(streak_region, (streak_region.shape[1]*4, streak_region.shape[0]*4))
    stages_vis.append(region_resized)

    # Blue mask
    blue_mask_colored = cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR)
    blue_mask_resized = cv2.resize(blue_mask_colored, (region_resized.shape[1], region_resized.shape[0]))
    stages_vis.append(blue_mask_resized)

    # Red mask
    red_mask_colored = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
    red_mask_resized = cv2.resize(red_mask_colored, (region_resized.shape[1], region_resized.shape[0]))
    stages_vis.append(red_mask_resized)

    # Final processed
    processed_colored = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    processed_resized = cv2.resize(processed_colored, (region_resized.shape[1], region_resized.shape[0]))
    stages_vis.append(processed_resized)

    # Stack stages horizontally
    stages_combined = np.hstack(stages_vis)

    # Add labels
    labels = ["Original", "Blue Mask (Loss)", "Red Mask (Win)", "OCR Processed"]
    labeled = stages_combined.copy()
    for i, label in enumerate(labels):
        x = i * region_resized.shape[1] + 10
        cv2.putText(labeled, label, (x, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imwrite(os.path.join(output_dir, "7_processing_stages.png"), labeled)

    # 9. Create summary image
    summary = np.zeros((800, 1200, 3), dtype=np.uint8)

    # Title
    cv2.putText(summary, "TFT Streak Detector Visualization", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 128, 0), 3)

    # Region info
    y_offset = 120
    info_lines = [
        f"Streak Position: {detector.streak_region['x1_rel']:.1%}-{detector.streak_region['x2_rel']:.1%} X, {detector.streak_region['y1_rel']:.1%}-{detector.streak_region['y2_rel']:.1%} Y",
        f"Pixel Coordinates: ({x1}, {y1}) to ({x2}, {y2})",
        f"Region Size: {x2-x1} x {y2-y1} pixels",
        f"Blue Range (Loss): H[{detector.blue_lower[0]}-{detector.blue_upper[0]}], S[{detector.blue_lower[1]}-{detector.blue_upper[1]}], V[{detector.blue_lower[2]}-{detector.blue_upper[2]}]",
        f"Red Range (Win): H[0-10, 170-180], S[50-255], V[50-255]",
    ]

    for line in info_lines:
        cv2.putText(summary, line, (50, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 35

    # Add small versions of key images
    # Original region
    region_small = cv2.resize(streak_region, (230, 66))
    summary[350:416, 50:280] = region_small
    cv2.putText(summary, "Extracted Region", (50, 340),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Processed
    processed_small = cv2.resize(cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR), (230, 66))
    summary[350:416, 320:550] = processed_small
    cv2.putText(summary, "Processed for OCR", (320, 340),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Detection results
    y_offset = 500
    cv2.putText(summary, "Detection Results:", (50, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2)
    y_offset += 40

    for screenshot_path in test_screenshots[:3]:
        if os.path.exists(screenshot_path):
            frame = cv2.imread(screenshot_path)
            if frame is not None:
                result = detector.detect_streak(frame)
                name = os.path.basename(screenshot_path).replace('.png', '')
                result_text = f"{name}: {result['type'].value} streak, length={result['length']}"
                cv2.putText(summary, result_text, (70, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_offset += 30

    cv2.imwrite(os.path.join(output_dir, "0_summary.png"), summary)

    print(f"\nVisualization files created in {output_dir}/:")
    for file in sorted(os.listdir(output_dir)):
        if file.endswith('.png'):
            print(f"  - {file}")

    # Test the detector
    print("\nStreak Detection Test Results:")
    for screenshot_path in test_screenshots:
        if os.path.exists(screenshot_path):
            frame = cv2.imread(screenshot_path)
            if frame is not None:
                result = detector.detect_streak(frame)
                name = os.path.basename(screenshot_path)
                print(f"\n{name}:")
                print(f"  Type: {result['type'].value}")
                print(f"  Length: {result['length']}")
                print(f"  Confidence: {result['confidence']:.2f}")

if __name__ == "__main__":
    create_streak_visualizations()