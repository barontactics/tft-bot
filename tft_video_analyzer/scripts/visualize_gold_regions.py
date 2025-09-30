#!/usr/bin/env python3
"""Visualize gold detection regions and processing steps"""

import cv2
import numpy as np
import os
from detectors.gold import TFTGoldDetector

def create_gold_visualizations():
    """Create comprehensive visualizations of gold detection"""

    detector = TFTGoldDetector()
    output_dir = "detectors/gold/gold_region"

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

    # 1. Full frame with gold region highlighted
    vis_full = test_frame.copy()
    x1 = int(w * detector.gold_region['x1_rel'])
    x2 = int(w * detector.gold_region['x2_rel'])
    y1 = int(h * detector.gold_region['y1_rel'])
    y2 = int(h * detector.gold_region['y2_rel'])

    cv2.rectangle(vis_full, (x1, y1), (x2, y2), (255, 215, 0), 3)  # Gold color
    cv2.putText(vis_full, "Gold Region", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 215, 0), 2)

    # Add coordinate info
    coord_text = f"({detector.gold_region['x1_rel']:.1%}, {detector.gold_region['y1_rel']:.1%}) to ({detector.gold_region['x2_rel']:.1%}, {detector.gold_region['y2_rel']:.1%})"
    cv2.putText(vis_full, coord_text, (x1, y2+30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 215, 0), 2)

    cv2.imwrite(os.path.join(output_dir, "1_full_frame_with_region.png"), vis_full)

    # 2. Extracted gold region
    gold_region = detector.extract_gold_region(test_frame)
    cv2.imwrite(os.path.join(output_dir, "2_extracted_region.png"), gold_region)

    # 3. Color mask visualization
    hsv = cv2.cvtColor(gold_region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, detector.gold_color_lower, detector.gold_color_upper)
    cv2.imwrite(os.path.join(output_dir, "3_color_mask.png"), mask)

    # 4. Masked region (only gold pixels)
    masked = cv2.bitwise_and(gold_region, gold_region, mask=mask)
    cv2.imwrite(os.path.join(output_dir, "4_masked_gold_pixels.png"), masked)

    # 5. Preprocessed for OCR
    processed = detector.preprocess_for_ocr(gold_region)
    cv2.imwrite(os.path.join(output_dir, "5_preprocessed_for_ocr.png"), processed)

    # 6. Process multiple screenshots and create a grid
    grid_images = []
    for i, screenshot_path in enumerate(test_screenshots[:4]):
        if os.path.exists(screenshot_path):
            frame = cv2.imread(screenshot_path)
            if frame is not None:
                # Draw gold region
                vis = frame.copy()
                h, w = frame.shape[:2]
                x1 = int(w * detector.gold_region['x1_rel'])
                x2 = int(w * detector.gold_region['x2_rel'])
                y1 = int(h * detector.gold_region['y1_rel'])
                y2 = int(h * detector.gold_region['y2_rel'])

                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 215, 0), 3)

                # Detect gold
                gold = detector.detect_gold(frame)
                gold_text = f"Gold: {gold}" if gold else "Gold: Not detected"
                cv2.putText(vis, gold_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 215, 0), 2)

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
    region_resized = cv2.resize(gold_region, (gold_region.shape[1]*4, gold_region.shape[0]*4))
    stages_vis.append(region_resized)

    # Grayscale
    gray = cv2.cvtColor(gold_region, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (gray.shape[1]*4, gray.shape[0]*4))
    gray_colored = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)
    stages_vis.append(gray_colored)

    # Binary threshold
    _, binary = cv2.threshold(gray_resized, 160, 255, cv2.THRESH_BINARY)
    binary_colored = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    stages_vis.append(binary_colored)

    # Final processed
    processed_colored = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    processed_resized = cv2.resize(processed_colored, (region_resized.shape[1], region_resized.shape[0]))
    stages_vis.append(processed_resized)

    # Stack stages horizontally
    stages_combined = np.hstack(stages_vis)

    # Add labels
    labels = ["Original", "Grayscale", "Binary", "Final"]
    labeled = stages_combined.copy()
    for i, label in enumerate(labels):
        x = i * region_resized.shape[1] + 10
        cv2.putText(labeled, label, (x, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imwrite(os.path.join(output_dir, "7_processing_stages.png"), labeled)

    # 9. Create summary image
    summary = np.zeros((800, 1200, 3), dtype=np.uint8)

    # Title
    cv2.putText(summary, "TFT Gold Detector Visualization", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 215, 0), 3)

    # Region info
    y_offset = 120
    info_lines = [
        f"Region Position: {detector.gold_region['x1_rel']:.1%}-{detector.gold_region['x2_rel']:.1%} X, {detector.gold_region['y1_rel']:.1%}-{detector.gold_region['y2_rel']:.1%} Y",
        f"Pixel Coordinates: ({x1}, {y1}) to ({x2}, {y2})",
        f"Region Size: {x2-x1} x {y2-y1} pixels",
        f"Color Range (HSV): H[{detector.gold_color_lower[0]}-{detector.gold_color_upper[0]}], S[{detector.gold_color_lower[1]}-{detector.gold_color_upper[1]}], V[{detector.gold_color_lower[2]}-{detector.gold_color_upper[2]}]",
    ]

    for line in info_lines:
        cv2.putText(summary, line, (50, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 40

    # Add small versions of key images
    # Original region
    region_small = cv2.resize(gold_region, (230, 66))
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
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 215, 0), 2)
    y_offset += 40

    for screenshot_path in test_screenshots[:3]:
        if os.path.exists(screenshot_path):
            frame = cv2.imread(screenshot_path)
            if frame is not None:
                gold = detector.detect_gold(frame)
                name = os.path.basename(screenshot_path).replace('.png', '')
                result_text = f"{name}: {gold if gold else 'Not detected'}"
                cv2.putText(summary, result_text, (70, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                y_offset += 30

    cv2.imwrite(os.path.join(output_dir, "0_summary.png"), summary)

    print(f"\nVisualization files created in {output_dir}/:")
    for file in sorted(os.listdir(output_dir)):
        if file.endswith('.png'):
            print(f"  - {file}")

if __name__ == "__main__":
    create_gold_visualizations()