#!/usr/bin/env python3
"""Test gold OCR on specific screenshot"""

import cv2
import numpy as np
import pytesseract
from detectors.gold import TFTGoldDetector

# Test on a screenshot
detector = TFTGoldDetector()
frame = cv2.imread("screenshots/round_1-3_frame03960_time01m06s.png")

if frame is not None:
    # Extract and process gold region
    gold_region = detector.extract_gold_region(frame)
    processed = detector.preprocess_for_ocr(gold_region)

    # Save for inspection
    cv2.imwrite("gold_extracted.png", gold_region)
    cv2.imwrite("gold_processed.png", processed)

    print("Gold region extracted and saved")
    print(f"Region shape: {gold_region.shape}")
    print(f"Processed shape: {processed.shape}")

    # Try different OCR configs
    configs = [
        ('PSM 8 (single word)', '--psm 8 -c tessedit_char_whitelist=0123456789'),
        ('PSM 7 (single line)', '--psm 7 -c tessedit_char_whitelist=0123456789'),
        ('PSM 13 (raw line)', '--psm 13 -c tessedit_char_whitelist=0123456789'),
        ('PSM 8 no whitelist', '--psm 8'),
        ('PSM 7 no whitelist', '--psm 7'),
    ]

    print("\nOCR Results:")
    for name, config in configs:
        try:
            text = pytesseract.image_to_string(processed, config=config).strip()
            print(f"  {name}: '{text}'")
        except Exception as e:
            print(f"  {name}: Error - {e}")

    # Also detect using the detector
    gold = detector.detect_gold(frame)
    print(f"\nDetector result: {gold}")

    # Check pixel statistics
    hsv = cv2.cvtColor(gold_region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, detector.gold_color_lower, detector.gold_color_upper)
    gold_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    print(f"\nGold pixels: {gold_pixels}/{total_pixels} ({gold_pixels/total_pixels*100:.1f}%)")