#!/usr/bin/env python3
"""Find the exact gold region by analyzing the screenshot"""

import cv2
import numpy as np

# Load a screenshot with visible gold
frame = cv2.imread("screenshots/round_1-3_frame03960_time01m06s.png")

if frame is not None:
    h, w = frame.shape[:2]
    print(f"Image dimensions: {w}x{h}")

    # The gold appears to be in the bottom center area
    # Let's check different regions
    regions = [
        ("Bottom-left", 0.05, 0.15, 0.87, 0.92),
        ("Bottom-center-left", 0.40, 0.50, 0.87, 0.92),
        ("Bottom-center", 0.45, 0.55, 0.85, 0.95),
        ("Bottom-UI", 0.42, 0.52, 0.88, 0.93),
        ("Gold-area", 0.44, 0.49, 0.89, 0.92),  # More precise
    ]

    for name, x1_rel, x2_rel, y1_rel, y2_rel in regions:
        x1 = int(w * x1_rel)
        x2 = int(w * x2_rel)
        y1 = int(h * y1_rel)
        y2 = int(h * y2_rel)

        region = frame[y1:y2, x1:x2]

        # Check for gold color
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        lower = np.array([20, 100, 100])
        upper = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        gold_ratio = np.sum(mask > 0) / mask.size

        print(f"\n{name} ({x1},{y1})-({x2},{y2}):")
        print(f"  Region size: {x2-x1}x{y2-y1}")
        print(f"  Gold pixel ratio: {gold_ratio:.3f}")

        # Save region for inspection
        cv2.imwrite(f"test_region_{name.lower().replace('-','_')}.png", region)

        # If gold ratio is good, try OCR
        if gold_ratio > 0.01:
            # Create binary image of gold pixels
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            result = cv2.bitwise_and(gray, gray, mask=mask)
            cv2.imwrite(f"test_mask_{name.lower().replace('-','_')}.png", result)

    # Also save full visualization
    vis_frame = frame.copy()
    # Draw the most likely gold region (bottom center)
    x1, x2 = int(w * 0.44), int(w * 0.49)
    y1, y2 = int(h * 0.89), int(h * 0.92)
    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 215, 0), 2)
    cv2.putText(vis_frame, "Gold Region", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 215, 0), 1)
    cv2.imwrite("gold_region_search.png", vis_frame)