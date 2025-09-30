#!/usr/bin/env python3
"""Final labeling script with all detectors working correctly"""

import cv2
import numpy as np
import os
import glob
import shutil
import pytesseract
import sys

# Add detectors to path if running standalone
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def detect_transition(frame):
    """Detect transition screens (mostly black)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    black_pixels = np.sum(gray < 10)
    black_ratio = black_pixels / gray.size
    return black_ratio > 0.80

def is_loading_screen(frame):
    """Detect loading screen with champion cards"""
    h, w = frame.shape[:2]

    # Loading screens have high purple content (8 cards spread across screen)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([130, 30, 30])
    upper = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    purple_ratio = np.sum(mask > 0) / mask.size

    # Loading screens have >40% purple content
    if purple_ratio > 0.40:
        # Check text region - loading screens won't have clean "Choose One"
        text_x1, text_x2 = int(w * 0.42), int(w * 0.58)
        text_y1, text_y2 = int(h * 0.18), int(h * 0.23)
        text_region = frame[text_y1:text_y2, text_x1:text_x2]

        text_gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
        _, text_binary = cv2.threshold(text_gray, 200, 255, cv2.THRESH_BINARY)
        white_ratio = np.sum(text_binary > 0) / text_binary.size

        # Loading screens: high purple + either no white text OR too much scattered text
        num_labels, _ = cv2.connectedComponents(text_binary)

        # Loading screen indicators:
        # - Very little white text (< 10%) OR
        # - Too many components (> 20, indicating scattered text/noise)
        if white_ratio < 0.10 or num_labels > 20:
            return True

    return False

def detect_augment_strict(frame):
    """Strict augment detection - must have Choose One text pattern"""
    if detect_transition(frame):
        return False

    # Reject if it's a loading screen
    if is_loading_screen(frame):
        return False

    h, w = frame.shape[:2]

    # Check text region for clean "Choose One" pattern
    text_x1, text_x2 = int(w * 0.42), int(w * 0.58)
    text_y1, text_y2 = int(h * 0.18), int(h * 0.23)
    text_region = frame[text_y1:text_y2, text_x1:text_x2]

    text_gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
    _, text_binary = cv2.threshold(text_gray, 200, 255, cv2.THRESH_BINARY)
    white_ratio = np.sum(text_binary > 0) / text_binary.size

    # "Choose One" text should be 10-30% white pixels
    if not (0.10 <= white_ratio <= 0.30):
        return False

    # Check connected components - "Choose One" has 8-15 components
    num_labels, _ = cv2.connectedComponents(text_binary)
    if not (8 <= num_labels <= 15):
        return False

    # Also check overall purple content - augments have less purple than loading
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([130, 30, 30])
    upper = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    overall_purple = np.sum(mask > 0) / mask.size

    # Augment screens have 20-35% purple (3 cards)
    # Loading screens have >40% purple (8 cards)
    if overall_purple > 0.35:
        return False

    # Verify 3 cards in augment positions
    card_width = int(w * 0.18)
    card_height = int(h * 0.42)
    card_y_center = int(h * 0.52)

    purple_count = 0
    card_positions = [0.295, 0.5, 0.705]

    for x_rel in card_positions:
        x_center = int(w * x_rel)
        x1 = max(0, x_center - card_width // 2)
        x2 = min(w, x_center + card_width // 2)
        y1 = max(0, card_y_center - card_height // 2)
        y2 = min(h, card_y_center + card_height // 2)

        card_region = frame[y1:y2, x1:x2]
        card_hsv = cv2.cvtColor(card_region, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(card_hsv, lower, upper)
        purple_ratio = np.sum(mask > 0) / mask.size

        if purple_ratio > 0.15:
            purple_count += 1

    # Must have exactly 3 purple cards
    return purple_count == 3

def detect_loading(frame):
    """Loading screen detection"""
    return is_loading_screen(frame)

def detect_planning_phase(frame):
    """Planning phase detection - yellow timer in top-left"""
    h, w = frame.shape[:2]
    timer_center_x = int(w * 0.095)
    timer_center_y = int(h * 0.048)
    timer_radius = int(min(w, h) * 0.025)

    timer_x1 = max(0, timer_center_x - timer_radius)
    timer_x2 = min(w, timer_center_x + timer_radius)
    timer_y1 = max(0, timer_center_y - timer_radius)
    timer_y2 = min(h, timer_center_y + timer_radius)

    timer_region = frame[timer_y1:timer_y2, timer_x1:timer_x2]

    if timer_region.size == 0:
        return False

    hsv = cv2.cvtColor(timer_region, cv2.COLOR_BGR2HSV)
    lower = np.array([20, 100, 100])
    upper = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    yellow_ratio = np.sum(mask > 0) / mask.size
    return yellow_ratio > 0.10

def detect_shop(frame):
    """Shop/buying phase detection"""
    h, w = frame.shape[:2]

    bottom_y = int(h * 0.75)
    bottom_region = frame[bottom_y:, :]

    gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size

    if edge_ratio > 0.15:
        hsv = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2HSV)
        # Gold color range
        lower_gold = np.array([20, 100, 100])
        upper_gold = np.array([30, 255, 255])
        gold_mask = cv2.inRange(hsv, lower_gold, upper_gold)
        gold_ratio = np.sum(gold_mask > 0) / gold_mask.size

        return gold_ratio > 0.01

    return False

def detect_combat(frame):
    """Combat phase detection"""
    h, w = frame.shape[:2]

    mid_region = frame[int(h*0.3):int(h*0.7), :]

    hsv = cv2.cvtColor(mid_region, cv2.COLOR_BGR2HSV)

    # Red health bars
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    red_ratio = np.sum(red_mask > 0) / red_mask.size

    # Green health bars
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(green_mask > 0) / green_mask.size

    return (red_ratio > 0.005 or green_ratio > 0.005) and not detect_shop(frame)

def detect_round_fast(frame):
    """Fast round detection with PSM 7"""
    h, w = frame.shape[:2]

    x1_rel, x2_rel = 0.403, 0.454
    y1_rel, y2_rel = 0.008, 0.03

    x1, x2 = int(w * x1_rel), int(w * x2_rel)
    y1, y2 = int(h * y1_rel), int(h * y2_rel)

    roi = frame[y1:y2, x1:x2].copy()

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    enhanced = clahe.apply(scaled)

    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        text = pytesseract.image_to_string(binary, config='--psm 7 -c tessedit_char_whitelist=0123456789-')
        text = text.strip()

        if '-' in text:
            parts = text.split('-')
            if len(parts) == 2:
                try:
                    stage = int(parts[0])
                    round_num = int(parts[1])
                    if 1 <= stage <= 10 and 1 <= round_num <= 10:
                        return f"{stage}-{round_num}"
                except:
                    pass
    except:
        pass

    return None

# Main processing
def main():
    screenshots_dir = "screenshots"
    output_dir = "screenshots_labeled"

    # Create fresh output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Get all screenshots
    screenshot_files = sorted(glob.glob(os.path.join(screenshots_dir, "*.png")))
    print(f"Found {len(screenshot_files)} screenshots to process\n")

    stats = {
        'loading': 0,
        'augment': 0,
        'planning': 0,
        'shop': 0,
        'combat': 0,
        'transition': 0,
        'unknown': 0,
        'with_round': 0,
        'without_round': 0
    }

    augment_files = []
    loading_files = []

    print("Processing screenshots with final detection logic...")
    print("Loading: >40% purple without 'Choose One' text pattern")
    print("Augment: 20-35% purple + clean 'Choose One' text + 3 cards\n")

    for i, filepath in enumerate(screenshot_files, 1):
        filename = os.path.basename(filepath)
        frame = cv2.imread(filepath)

        if frame is None:
            print(f"  [{i:3d}] Error loading {filename}")
            continue

        # Detect state - ORDER MATTERS!
        state = "unknown"

        if detect_transition(frame):
            state = "transition"
        elif detect_loading(frame):
            state = "loading"
            loading_files.append(filename)
        elif detect_augment_strict(frame):
            state = "augment"
            augment_files.append(filename)
        elif detect_planning_phase(frame):
            state = "planning"
        elif detect_shop(frame):
            state = "shop"
        elif detect_combat(frame):
            state = "combat"

        # Detect round
        round_info = detect_round_fast(frame)

        # Update stats
        stats[state] += 1
        if round_info:
            stats['with_round'] += 1
        else:
            stats['without_round'] += 1

        # Create new filename
        base_name = filename.replace('.png', '')
        if round_info:
            round_str = f"round{round_info.replace('-', '_')}"
        else:
            round_str = "noround"

        new_filename = f"{state}_{round_str}_{base_name}.png"
        new_filepath = os.path.join(output_dir, new_filename)

        # Copy file (no duplicates)
        shutil.copy2(filepath, new_filepath)

        if i % 20 == 0:
            print(f"  [{i:3d}/{len(screenshot_files)}] Processed {i} files...")

    print("\n" + "="*60)
    print("FINAL LABELING COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir}/")
    print("\nDetection Statistics:")
    for key, value in stats.items():
        if key not in ['with_round', 'without_round']:
            print(f"  {key:12}: {value:4d} screenshots")

    print(f"\nRound detection:")
    print(f"  With round:    {stats['with_round']:4d} screenshots")
    print(f"  Without round: {stats['without_round']:4d} screenshots")
    print(f"  Success rate:  {stats['with_round']/len(screenshot_files)*100:.1f}%")

    print(f"\nKey counts:")
    print(f"  Loading screens: {stats['loading']} (>40% purple, no 'Choose One')")
    print(f"  Augment screens: {stats['augment']} (20-35% purple, has 'Choose One')")

    # Check frames 022-028 specifically
    print("\nFrames 00000-00900 classification:")
    for filename in screenshot_files:
        if any(f"frame{num:05d}" in filename for num in range(0, 1081, 180)):
            base = os.path.basename(filename)
            matches = [f for f in os.listdir(output_dir) if base in f]
            if matches:
                state = matches[0].split('_')[0]
                print(f"  {base} -> {state.upper()}")

if __name__ == "__main__":
    main()