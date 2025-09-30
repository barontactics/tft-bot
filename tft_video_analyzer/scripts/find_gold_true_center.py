#!/usr/bin/env python3
"""Find gold in the TRUE center of the screen"""

import cv2
import numpy as np

frame = cv2.imread("screenshots/round_1-3_frame03960_time01m06s.png")

if frame is not None:
    h, w = frame.shape[:2]
    print(f"Image dimensions: {w}x{h}")
    print(f"Screen center X: {w//2} (50%)")
    print(f"Screen center Y: {h//2} (50%)")
    print("\nSearching for gold in TRUE CENTER of screen...\n")

    # Current region is 6% wide, so to center it we need 47-53%
    # Height is correct at 81.5-84.5%
    regions = [
        ("True-Center", 0.47, 0.53, 0.815, 0.845),     # Exactly centered
        ("Center-Slightly-Left", 0.46, 0.52, 0.815, 0.845),
        ("Center-Slightly-Right", 0.48, 0.54, 0.815, 0.845),
        ("Wider-Center", 0.45, 0.55, 0.815, 0.845),    # 10% wide
        ("Narrower-Center", 0.48, 0.52, 0.815, 0.845), # 4% wide
        ("Current-Wrong", 0.42, 0.48, 0.815, 0.845),   # Current position (off-center)
    ]

    best_ratio = 0
    best_region = None

    for name, x1_rel, x2_rel, y1_rel, y2_rel in regions:
        x1 = int(w * x1_rel)
        x2 = int(w * x2_rel)
        y1 = int(h * y1_rel)
        y2 = int(h * y2_rel)

        # Calculate center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        offset_from_center = center_x - w//2

        region = frame[y1:y2, x1:x2]

        # Check for gold/yellow color
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        lower = np.array([10, 80, 150])
        upper = np.array([25, 200, 255])
        mask = cv2.inRange(hsv, lower, upper)
        gold_ratio = np.sum(mask > 0) / mask.size

        print(f"{name}:")
        print(f"  Position: {x1_rel:.1%}-{x2_rel:.1%} x, {y1_rel:.1%}-{y2_rel:.1%} y")
        print(f"  Center: ({center_x}, {center_y})")
        print(f"  Offset from screen center: {offset_from_center:+d} pixels")
        print(f"  Gold pixel ratio: {gold_ratio:.4f}")

        if gold_ratio > best_ratio:
            best_ratio = gold_ratio
            best_region = (name, x1_rel, x2_rel, y1_rel, y2_rel, x1, y1, x2, y2)

        # Save regions with gold pixels
        if gold_ratio > 0.001:
            cv2.imwrite(f"gold_true_center_{name.lower().replace('-','_').replace(' ','_')}.png", region)

        print()

    if best_region:
        print(f"Best region found: {best_region[0]}")
        print(f"  Position: {best_region[1]:.1%}-{best_region[2]:.1%} x")
        print(f"  Gold ratio: {best_ratio:.4f}")

    # Create visualization
    vis_frame = frame.copy()

    # Draw center lines
    cv2.line(vis_frame, (w//2, 0), (w//2, h), (0, 255, 255), 1)  # Vertical center line
    cv2.line(vis_frame, (0, h//2), (w, h//2), (0, 255, 255), 1)  # Horizontal center line
    cv2.putText(vis_frame, "Screen Center", (w//2 + 10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]

    for i, (name, x1_rel, x2_rel, y1_rel, y2_rel) in enumerate(regions):
        x1 = int(w * x1_rel)
        x2 = int(w * x2_rel)
        y1 = int(h * y1_rel)
        y2 = int(h * y2_rel)
        color = colors[i % len(colors)]

        # Draw rectangle
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

        # Draw center point
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(vis_frame, (cx, cy), 3, color, -1)

        # Label
        cv2.putText(vis_frame, name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Highlight best region
    if best_region:
        cv2.rectangle(vis_frame, (best_region[5], best_region[6]),
                     (best_region[7], best_region[8]), (0, 255, 0), 3)

    cv2.imwrite("gold_search_true_center.png", vis_frame)
    print("\nVisualization saved to gold_search_true_center.png")