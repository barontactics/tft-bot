#!/usr/bin/env python3
"""
TFT Planning Phase Detector
Detects the start of planning phase by looking for "Planning" text in the middle of the screen
"""

import cv2
import numpy as np
import pytesseract
from typing import Optional, Tuple

class TFTPlanningDetector:
    """Detects the start of planning phase in TFT"""

    def __init__(self):
        # Region where "Planning" text appears (center of screen)
        self.planning_text_region = (0.35, 0.45, 0.65, 0.55)  # Center area

        # Alternative regions to check if text moves
        self.alt_regions = [
            (0.40, 0.40, 0.60, 0.50),  # Slightly higher
            (0.30, 0.45, 0.70, 0.55),  # Wider area
            (0.35, 0.48, 0.65, 0.58),  # Slightly lower
        ]

    def extract_region(self, frame: np.ndarray, region: Tuple[float, float, float, float]) -> np.ndarray:
        """Extract a region from the frame using relative coordinates"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = region
        x1, x2 = int(x1 * w), int(x2 * w)
        y1, y2 = int(y1 * h), int(y2 * h)
        return frame[y1:y2, x1:x2]

    def preprocess_for_text(self, image: np.ndarray) -> list:
        """Preprocess image for better OCR text detection"""
        preprocessed = []

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Method 1: Binary threshold for white text
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        preprocessed.append(binary)

        # Method 2: Binary threshold with lower threshold
        _, binary2 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        preprocessed.append(binary2)

        # Method 3: Inverted binary for dark text
        _, binary_inv = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        preprocessed.append(binary_inv)

        # Method 4: Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        preprocessed.append(adaptive)

        # Method 5: Scale up for better OCR
        scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, scaled_binary = cv2.threshold(scaled, 200, 255, cv2.THRESH_BINARY)
        preprocessed.append(scaled_binary)

        return preprocessed

    def detect_planning_text(self, region: np.ndarray) -> bool:
        """Detect if 'Planning' text is present in the region"""

        # Get preprocessed versions
        preprocessed_images = self.preprocess_for_text(region)

        # Try OCR on each preprocessed version
        for processed in preprocessed_images:
            try:
                # Use OCR to detect text
                text = pytesseract.image_to_string(
                    processed,
                    config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
                )

                # Clean and check text
                text_clean = text.strip().lower()

                # Check for "planning" keyword
                if 'planning' in text_clean:
                    return True

                # Check for partial matches (in case OCR is imperfect)
                if 'plann' in text_clean or 'lanning' in text_clean:
                    return True

            except Exception as e:
                continue

        return False

    def detect_planning_phase(self, frame: np.ndarray) -> bool:
        """
        Detect if the frame shows the start of planning phase

        Args:
            frame: OpenCV image (numpy array)

        Returns:
            True if planning phase detected, False otherwise
        """

        # Check main region first
        main_region = self.extract_region(frame, self.planning_text_region)

        if self.detect_planning_text(main_region):
            return True

        # Check alternative regions if main region fails
        for alt_region in self.alt_regions:
            region = self.extract_region(frame, alt_region)
            if self.detect_planning_text(region):
                return True

        # Additional check: Look for UI elements that appear during planning
        # Planning phase typically has bright center area when text appears
        gray = cv2.cvtColor(main_region, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        # If center is very bright, might be planning text (white on darker background)
        if mean_brightness > 200:
            # Double-check with less strict text detection
            text = pytesseract.image_to_string(main_region, config='--psm 7')
            if any(word in text.lower() for word in ['plan', 'ning', 'lann']):
                return True

        return False

    def get_debug_info(self, frame: np.ndarray) -> dict:
        """Get detailed debug information about planning detection"""
        info = {
            'is_planning': False,
            'regions_checked': [],
            'text_found': []
        }

        # Check main region
        main_region = self.extract_region(frame, self.planning_text_region)

        try:
            text = pytesseract.image_to_string(main_region, config='--psm 7')
            info['text_found'].append(('main', text.strip()))
        except:
            info['text_found'].append(('main', 'OCR_FAILED'))

        if self.detect_planning_text(main_region):
            info['is_planning'] = True
            info['regions_checked'].append('main')

        # Check alternatives if not found
        if not info['is_planning']:
            for i, alt_region in enumerate(self.alt_regions):
                region = self.extract_region(frame, alt_region)
                if self.detect_planning_text(region):
                    info['is_planning'] = True
                    info['regions_checked'].append(f'alt_{i}')
                    break

        return info