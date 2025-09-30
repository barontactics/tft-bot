#!/usr/bin/env python3
"""Search for gold in higher UI areas"""

import cv2
import numpy as np

frame = cv2.imread("screenshots/round_1-3_frame03960_time01m06s.png")

if frame is not None:
    h, w = frame.shape[:2]
    print(f"Image dimensions: {w}x{h}")
    print("Searching for gold in higher UI areas...\n")

    # Gold is typically displayed above the shop area, not in it
    # Check various positions higher up in the bottom UI
    regions = [
        ("Bottom-bar-left", 0.10, 0.25, 0.82, 0.88),     # Left side of bottom bar
        ("Bottom-bar-center", 0.40, 0.60, 0.82, 0.88),   # Center of bottom bar
        ("Above-shop-left", 0.10, 0.20, 0.85, 0.90),     # Above shop, left
        ("Above-shop-center", 0.45, 0.55, 0.83, 0.89),   # Above shop, center
        ("UI-bar-gold", 0.08, 0.15, 0.84, 0.87),         # Common gold position
        ("Mid-bottom-left", 0.05, 0.15, 0.80, 0.85),     # Higher up, left
        ("Gold-icon-area", 0.07, 0.13, 0.835, 0.865),    # Precise gold icon area
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
        # Gold/yellow color range
        lower = np.array([15, 100, 100])
        upper = np.array([35, 255, 255])
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
            cv2.imwrite(f"gold_region_{name.lower().replace('-','_').replace(' ','_')}.png", region)

            # Also save processed version
            result = cv2.bitwise_and(region, region, mask=mask)
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(f"gold_mask_{name.lower().replace('-','_').replace(' ','_')}.png", gray)

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

    cv2.imwrite("gold_search_higher.png", vis_frame)
    print("\nVisualization saved to gold_search_higher.png")