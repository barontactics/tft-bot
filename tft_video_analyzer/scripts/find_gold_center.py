#!/usr/bin/env python3
"""Find gold in the center of the screen"""

import cv2
import numpy as np

frame = cv2.imread("screenshots/round_1-3_frame03960_time01m06s.png")

if frame is not None:
    h, w = frame.shape[:2]
    print(f"Image dimensions: {w}x{h}")
    print("Searching for gold in CENTER of screen...\n")

    # Gold should be in center, 2% higher than current position
    # Current is 83.5-86.5%, so new should be 81.5-84.5%
    # Center of screen is around 45-55%
    regions = [
        ("Center-1", 0.45, 0.55, 0.815, 0.845),    # Center, 2% higher
        ("Center-2", 0.44, 0.56, 0.815, 0.845),    # Slightly wider
        ("Center-3", 0.46, 0.54, 0.815, 0.845),    # Slightly narrower
        ("Center-bottom", 0.45, 0.55, 0.825, 0.855), # Center, 1% higher
        ("Center-exact", 0.47, 0.53, 0.815, 0.845),  # More precise center
        ("Center-left", 0.42, 0.48, 0.815, 0.845),   # Center-left
        ("Center-right", 0.52, 0.58, 0.815, 0.845),  # Center-right
    ]

    best_ratio = 0
    best_region = None

    for name, x1_rel, x2_rel, y1_rel, y2_rel in regions:
        x1 = int(w * x1_rel)
        x2 = int(w * x2_rel)
        y1 = int(h * y1_rel)
        y2 = int(h * y2_rel)

        region = frame[y1:y2, x1:x2]

        # Check for gold/yellow color
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        # Gold/orange color range
        lower = np.array([10, 80, 150])
        upper = np.array([25, 200, 255])
        mask = cv2.inRange(hsv, lower, upper)
        gold_ratio = np.sum(mask > 0) / mask.size

        print(f"{name}:")
        print(f"  Position: {x1_rel:.1%}-{x2_rel:.1%} x, {y1_rel:.1%}-{y2_rel:.1%} y")
        print(f"  Coordinates: ({x1},{y1})-({x2},{y2})")
        print(f"  Gold pixel ratio: {gold_ratio:.4f}")

        if gold_ratio > best_ratio:
            best_ratio = gold_ratio
            best_region = (name, x1_rel, x2_rel, y1_rel, y2_rel, x1, y1, x2, y2)

        # Save regions with significant gold pixels
        if gold_ratio > 0.001:
            cv2.imwrite(f"gold_center_{name.lower().replace('-','_').replace(' ','_')}.png", region)

        print()

    if best_region:
        print(f"Best region found: {best_region[0]}")
        print(f"  Position: {best_region[1]:.1%}-{best_region[2]:.1%} x, {best_region[3]:.1%}-{best_region[4]:.1%} y")
        print(f"  Gold ratio: {best_ratio:.4f}")

    # Create visualization showing all search areas
    vis_frame = frame.copy()
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (255,128,0)]

    for i, (name, x1_rel, x2_rel, y1_rel, y2_rel) in enumerate(regions):
        x1 = int(w * x1_rel)
        x2 = int(w * x2_rel)
        y1 = int(h * y1_rel)
        y2 = int(h * y2_rel)
        color = colors[i % len(colors)]
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_frame, name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Highlight best region
    if best_region:
        cv2.rectangle(vis_frame, (best_region[5], best_region[6]),
                     (best_region[7], best_region[8]), (0, 255, 0), 3)

    # Also draw current incorrect region for comparison
    old_x1 = int(w * 0.07)
    old_x2 = int(w * 0.13)
    old_y1 = int(h * 0.835)
    old_y2 = int(h * 0.865)
    cv2.rectangle(vis_frame, (old_x1, old_y1), (old_x2, old_y2), (0, 0, 255), 2)
    cv2.putText(vis_frame, "OLD (wrong)", (old_x1, old_y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imwrite("gold_search_center.png", vis_frame)
    print("\nVisualization saved to gold_search_center.png")