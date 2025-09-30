#!/usr/bin/env python3
"""
TFT Loading Screen Detector
Detects the loading screen with player sprites at the start of a TFT game
"""

import cv2
import numpy as np
from typing import Tuple, Optional

class TFTLoadingScreenDetector:
    """Detects TFT loading screen with player sprites"""

    def __init__(self):
        # Player cards are in 2 rows, covering most of the center screen
        self.player_card_region = (0.15, 0.25, 0.85, 0.75)  # Covers both rows of cards

    def detect_loading_screen(self, frame: np.ndarray) -> bool:
        """
        Detect if the current frame is a loading screen
        Loading screen characteristics:
        - 8 player cards arranged in 2 rows (4 top, 4 bottom)
        - Purple/pink background with swirls
        - Cards have glowing borders
        - Player names below sprites
        """
        h, w = frame.shape[:2]

        # Check card region
        x1, y1, x2, y2 = self.player_card_region
        x1, x2 = int(x1 * w), int(x2 * w)
        y1, y2 = int(y1 * h), int(y2 * h)

        card_region = frame[y1:y2, x1:x2]

        # Convert to HSV for color detection
        hsv = cv2.cvtColor(card_region, cv2.COLOR_BGR2HSV)

        # Check for purple/pink colors (characteristic of loading screen)
        # Purple/pink in HSV: Hue around 280-320 (in OpenCV: 140-160)
        lower_purple = np.array([130, 30, 30])
        upper_purple = np.array([170, 255, 255])
        purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
        purple_ratio = np.sum(purple_mask > 0) / purple_mask.size

        # Check for card structures using edge detection
        gray = cv2.cvtColor(card_region, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect edges
        edges = cv2.Canny(blurred, 30, 100)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter for card-like rectangles
        card_count = 0
        min_card_area = 8000  # Minimum area for a player card
        max_card_area = 50000  # Maximum area to filter out too large contours

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_card_area < area < max_card_area:
                # Check if shape is roughly rectangular
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

                if 4 <= len(approx) <= 8:  # Rectangle-like shape (allowing for rounded corners)
                    x, y, w_cont, h_cont = cv2.boundingRect(contour)
                    aspect_ratio = h_cont / w_cont if w_cont > 0 else 0

                    # Player cards are taller than wide (portrait orientation)
                    if 1.2 <= aspect_ratio <= 2.0:
                        card_count += 1

        # Also check for high contrast areas (cards stand out from background)
        std_dev = np.std(gray)
        has_high_contrast = std_dev > 40

        # Loading screen detection criteria:
        # Loading screens have 8 player cards (4 top, 4 bottom)
        # Augment screens have 3 cards in one row

        # If we detect 2-4 cards, it's likely an augment screen, not loading
        if 2 <= card_count <= 4:
            return False

        # Loading screens need:
        # 1. Multiple cards (6-8 for all players)
        # 2. Purple/pink colors
        # 3. Dark background (loading screens are darker than augment screens)
        # 4. High contrast between cards and background
        is_loading = card_count >= 6 and purple_ratio > 0.05 and has_high_contrast

        # Additional darkness check to distinguish from augment screens
        # Loading screens have darker backgrounds
        mean_brightness = np.mean(cv2.cvtColor(card_region, cv2.COLOR_BGR2GRAY))
        if is_loading and mean_brightness > 100:
            # Too bright for a loading screen, might be augment
            is_loading = False

        return is_loading

    def get_loading_screen_info(self, frame: np.ndarray) -> dict:
        """Get detailed information about the loading screen"""
        is_loading = self.detect_loading_screen(frame)

        info = {
            'is_loading_screen': is_loading,
            'timestamp': None,  # To be filled by caller
        }

        if is_loading:
            # Could add player name detection here using OCR
            info['player_count'] = 8  # TFT always has 8 players

        return info