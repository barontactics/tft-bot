#!/usr/bin/env python3
"""Analyze the gold text color to find better thresholds"""

import cv2
import numpy as np
from detectors.gold import TFTGoldDetector

# Load the screenshot
detector = TFTGoldDetector()
frame = cv2.imread("screenshots/round_1-3_frame03960_time01m06s.png")

if frame is not None:
    # Extract gold region
    gold_region = detector.extract_gold_region(frame)

    # Convert to HSV
    hsv = cv2.cvtColor(gold_region, cv2.COLOR_BGR2HSV)

    # Get HSV values for all pixels
    h, w = gold_region.shape[:2]

    print("Analyzing gold region colors...")
    print(f"Region size: {w}x{h}")

    # Sample the center area where the number should be
    center_x = w // 2
    center_y = h // 2
    sample_radius = 10

    # Get pixels from center area
    center_region = gold_region[max(0, center_y-sample_radius):min(h, center_y+sample_radius),
                                max(0, center_x-sample_radius):min(w, center_x+sample_radius)]
    center_hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)

    # Find bright pixels (likely text)
    bright_mask = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY) > 100

    if np.any(bright_mask):
        bright_pixels_hsv = center_hsv[bright_mask]

        print(f"\nBright pixels in center: {len(bright_pixels_hsv)}")
        if len(bright_pixels_hsv) > 0:
            h_vals = bright_pixels_hsv[:, 0]
            s_vals = bright_pixels_hsv[:, 1]
            v_vals = bright_pixels_hsv[:, 2]

            print(f"Hue range: {h_vals.min()}-{h_vals.max()} (mean: {h_vals.mean():.1f})")
            print(f"Saturation range: {s_vals.min()}-{s_vals.max()} (mean: {s_vals.mean():.1f})")
            print(f"Value range: {v_vals.min()}-{v_vals.max()} (mean: {v_vals.mean():.1f})")

    # Try different color ranges
    test_ranges = [
        ("Original", np.array([20, 100, 100]), np.array([35, 255, 255])),
        ("Broader yellow", np.array([15, 50, 100]), np.array([40, 255, 255])),
        ("Very broad", np.array([10, 30, 80]), np.array([45, 255, 255])),
        ("White-yellow", np.array([15, 0, 150]), np.array([35, 100, 255])),
        ("Low saturation", np.array([20, 20, 150]), np.array([35, 150, 255])),
    ]

    print("\nTesting different color ranges:")
    for name, lower, upper in test_ranges:
        mask = cv2.inRange(hsv, lower, upper)
        pixel_count = np.sum(mask > 0)
        ratio = pixel_count / mask.size
        print(f"  {name}: {pixel_count} pixels ({ratio*100:.2f}%)")

        # Save mask for inspection
        cv2.imwrite(f"gold_mask_{name.lower().replace(' ', '_').replace('-', '_')}.png", mask)

    # Also try simple brightness threshold
    gray = cv2.cvtColor(gold_region, cv2.COLOR_BGR2GRAY)
    bright_thresholds = [100, 120, 140, 160, 180]

    print("\nBrightness thresholds:")
    for thresh in bright_thresholds:
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        pixel_count = np.sum(binary > 0)
        ratio = pixel_count / binary.size
        print(f"  Threshold {thresh}: {pixel_count} pixels ({ratio*100:.2f}%)")

        if thresh == 140:
            cv2.imwrite("gold_brightness_140.png", binary)

    # Save original for comparison
    cv2.imwrite("gold_original.png", gold_region)