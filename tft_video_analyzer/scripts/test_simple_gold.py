#!/usr/bin/env python3
"""Test simplified gold detection"""

import cv2
import numpy as np
import pytesseract

def detect_gold_simple(frame):
    """Simple gold detection using brightness threshold"""
    h, w = frame.shape[:2]

    # Gold region (7-13% from left, 83.5-86.5% from top)
    x1 = int(w * 0.07)
    x2 = int(w * 0.13)
    y1 = int(h * 0.835)
    y2 = int(h * 0.865)

    region = frame[y1:y2, x1:x2]

    # Convert to grayscale
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # Scale up 4x
    scaled = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # Simple threshold
    _, binary = cv2.threshold(scaled, 160, 255, cv2.THRESH_BINARY)

    # Try OCR with different configs
    configs = [
        '--psm 8 -c tessedit_char_whitelist=0123456789',
        '--psm 7 -c tessedit_char_whitelist=0123456789',
        '--psm 13 -c tessedit_char_whitelist=0123456789',
        '--psm 8',
        '-c tessedit_char_whitelist=0123456789',
    ]

    for config in configs:
        try:
            text = pytesseract.image_to_string(binary, config=config).strip()
            if text and text[0].isdigit():
                # Extract just the number
                num_str = ''.join(c for c in text if c.isdigit())
                if num_str:
                    gold = int(num_str)
                    if 0 <= gold <= 999:
                        return gold
        except:
            pass

    return None

# Test on screenshots
test_files = [
    "screenshots/round_1-3_frame03960_time01m06s.png",
    "screenshots/round_2-3_frame18720_time05m12s.png",
    "screenshots/round_2-2_frame16200_time04m30s.png",
]

for filepath in test_files:
    frame = cv2.imread(filepath)
    if frame is not None:
        gold = detect_gold_simple(frame)
        print(f"{filepath}: {gold if gold else 'Not detected'}")

# Save debug image for first file
frame = cv2.imread(test_files[0])
if frame is not None:
    h, w = frame.shape[:2]
    x1 = int(w * 0.07)
    x2 = int(w * 0.13)
    y1 = int(h * 0.835)
    y2 = int(h * 0.865)

    region = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(scaled, 160, 255, cv2.THRESH_BINARY)

    cv2.imwrite("gold_simple_binary.png", binary)
    print(f"\nBinary image saved to gold_simple_binary.png")