#!/usr/bin/env python3
"""Find gold region more precisely by examining bottom UI"""

import cv2
import numpy as np

frame = cv2.imread("screenshots/round_1-3_frame03960_time01m06s.png")

if frame is not None:
    h, w = frame.shape[:2]
    print(f"Image dimensions: {w}x{h}")

    # Gold appears at the bottom center, near the shop area
    # Looking at the image, it's around 47-48% from left, 91-93% from top
    # Try more precise regions around the visible "2"
    regions = [
        ("Gold-exact-1", 0.465, 0.495, 0.905, 0.925),  # Near center bottom
        ("Gold-exact-2", 0.470, 0.490, 0.910, 0.923),  # Slightly adjusted
        ("Gold-exact-3", 0.473, 0.487, 0.912, 0.922),  # More precise
        ("Gold-icon-area", 0.455, 0.475, 0.910, 0.923),  # Gold icon area
        ("Full-bottom", 0.40, 0.60, 0.90, 0.95),  # Wider search
    ]

    best_ratio = 0
    best_region = None

    for name, x1_rel, x2_rel, y1_rel, y2_rel in regions:
        x1 = int(w * x1_rel)
        x2 = int(w * x2_rel)
        y1 = int(h * y1_rel)
        y2 = int(h * y2_rel)

        region = frame[y1:y2, x1:x2]

        # Check for gold/yellow color with broader range
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        # Broader gold/yellow range
        lower = np.array([15, 50, 50])
        upper = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        gold_ratio = np.sum(mask > 0) / mask.size

        print(f"\n{name} ({x1},{y1})-({x2},{y2}):")
        print(f"  Position: {x1_rel:.1%}-{x2_rel:.1%} x, {y1_rel:.1%}-{y2_rel:.1%} y")
        print(f"  Region size: {x2-x1}x{y2-y1}")
        print(f"  Gold pixel ratio: {gold_ratio:.3f}")

        if gold_ratio > best_ratio:
            best_ratio = gold_ratio
            best_region = (name, x1_rel, x2_rel, y1_rel, y2_rel)

        # Save region for inspection
        cv2.imwrite(f"gold_search_{name.lower().replace('-','_')}.png", region)

        if gold_ratio > 0.01:
            # Process for OCR
            result = cv2.bitwise_and(region, region, mask=mask)
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            cv2.imwrite(f"gold_binary_{name.lower().replace('-','_')}.png", binary)

    print(f"\nBest region: {best_region[0] if best_region else 'None'}")
    print(f"Best ratio: {best_ratio:.3f}")

    # Visualize all regions
    vis_frame = frame.copy()
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]

    for i, (name, x1_rel, x2_rel, y1_rel, y2_rel) in enumerate(regions):
        x1 = int(w * x1_rel)
        x2 = int(w * x2_rel)
        y1 = int(h * y1_rel)
        y2 = int(h * y2_rel)
        color = colors[i % len(colors)]
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_frame, name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imwrite("gold_all_regions.png", vis_frame)
    print("\nVisualization saved to gold_all_regions.png")