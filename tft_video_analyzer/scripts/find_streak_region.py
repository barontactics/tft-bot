#!/usr/bin/env python3
"""Find the streak region to the right of gold"""

import cv2
import numpy as np

frame = cv2.imread("screenshots/round_1-3_frame03960_time01m06s.png")

if frame is not None:
    h, w = frame.shape[:2]
    print(f"Image dimensions: {w}x{h}")
    print(f"Gold region: 47%-53% x, 81.5%-84.5% y")
    print("Searching for streak region to the RIGHT of gold...\n")

    # Streak should be to the right of gold (53%+)
    # Same height as gold (81.5-84.5%)
    regions = [
        ("Right-of-gold-1", 0.53, 0.59, 0.815, 0.845),   # Directly right of gold
        ("Right-of-gold-2", 0.54, 0.60, 0.815, 0.845),   # Slightly more right
        ("Right-of-gold-3", 0.55, 0.61, 0.815, 0.845),   # Even more right
        ("Right-wider", 0.53, 0.62, 0.815, 0.845),       # Wider region
        ("Right-narrower", 0.54, 0.58, 0.815, 0.845),    # Narrower
        ("Right-lower", 0.53, 0.59, 0.825, 0.855),       # Slightly lower
        ("Right-higher", 0.53, 0.59, 0.805, 0.835),      # Slightly higher
    ]

    for name, x1_rel, x2_rel, y1_rel, y2_rel in regions:
        x1 = int(w * x1_rel)
        x2 = int(w * x2_rel)
        y1 = int(h * y1_rel)
        y2 = int(h * y2_rel)

        region = frame[y1:y2, x1:x2]

        # Check for blue (loss streak) and red (win streak) colors
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Blue fire range (loss streak)
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        blue_ratio = np.sum(blue_mask > 0) / blue_mask.size

        # Red/orange fire range (win streak)
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)

        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)

        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_ratio = np.sum(red_mask > 0) / red_mask.size

        # Check for any bright pixels (text/numbers)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        bright_pixels = np.sum(gray > 180)
        bright_ratio = bright_pixels / gray.size

        print(f"{name}:")
        print(f"  Position: {x1_rel:.1%}-{x2_rel:.1%} x, {y1_rel:.1%}-{y2_rel:.1%} y")
        print(f"  Coordinates: ({x1},{y1})-({x2},{y2})")
        print(f"  Blue pixels: {blue_ratio:.4f}")
        print(f"  Red pixels: {red_ratio:.4f}")
        print(f"  Bright pixels: {bright_ratio:.4f}")

        # Save region for inspection
        cv2.imwrite(f"streak_region_{name.lower().replace('-','_').replace(' ','_')}.png", region)

        # Save masks
        if blue_ratio > 0.001 or red_ratio > 0.001:
            cv2.imwrite(f"streak_blue_{name.lower().replace('-','_').replace(' ','_')}.png", blue_mask)
            cv2.imwrite(f"streak_red_{name.lower().replace('-','_').replace(' ','_')}.png", red_mask)

        print()

    # Create visualization
    vis_frame = frame.copy()

    # Draw gold region for reference
    gold_x1 = int(w * 0.47)
    gold_x2 = int(w * 0.53)
    gold_y1 = int(h * 0.815)
    gold_y2 = int(h * 0.845)
    cv2.rectangle(vis_frame, (gold_x1, gold_y1), (gold_x2, gold_y2), (255, 215, 0), 2)
    cv2.putText(vis_frame, "Gold", (gold_x1, gold_y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 215, 0), 1)

    # Draw streak search regions
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (128,255,128)]

    for i, (name, x1_rel, x2_rel, y1_rel, y2_rel) in enumerate(regions):
        x1 = int(w * x1_rel)
        x2 = int(w * x2_rel)
        y1 = int(h * y1_rel)
        y2 = int(h * y2_rel)
        color = colors[i % len(colors)]
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_frame, name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imwrite("streak_search_regions.png", vis_frame)
    print("Visualization saved to streak_search_regions.png")