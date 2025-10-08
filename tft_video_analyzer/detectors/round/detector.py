#!/usr/bin/env python3
"""
TFT Round Detector
Detects stage and round numbers from the top-left UI indicator (e.g., "1-2" for stage 1, round 2)
"""

import cv2
import numpy as np
import easyocr
from typing import Optional, Tuple
import re


class TFTRoundDetector:
    """Detects stage and round from TFT gameplay"""

    def __init__(self):
        # Initialize EasyOCR reader (English only for numbers and dash)
        self.reader = easyocr.Reader(['en'], gpu=False)

        # Region where stage-round indicator appears (top center of info bar)
        # Format is typically "STAGE-ROUND" like "1-2", "3-4", etc.
        # Based on EasyOCR scan: found at (0.4282, 0.0135, 0.4474, 0.0310)
        # NOTE: UI position shifts ~10% horizontally in different game phases
        self.round_region = (0.425, 0.012, 0.450, 0.032)  # Top center

        # Alternative regions to account for UI shifts across game phases
        # The UI can shift horizontally by ~10% during different phases
        self.alt_regions = [
            # Original position variations
            (0.420, 0.012, 0.455, 0.032),  # Slightly wider
            (0.425, 0.010, 0.450, 0.035),  # Slightly taller
            (0.428, 0.012, 0.448, 0.032),  # Exact from scan

            # Shifted left by ~5% for alternate UI layouts (moved right from previous -10%)
            (0.375, 0.012, 0.400, 0.032),  # Shifted left by 5%
            (0.370, 0.012, 0.405, 0.032),  # Shifted left by 5%, wider
            (0.375, 0.010, 0.400, 0.035),  # Shifted left by 5%, taller

            # Shifted right by ~5% for other layouts
            (0.475, 0.012, 0.500, 0.032),  # Shifted right
            (0.470, 0.012, 0.505, 0.032),  # Shifted right, wider

            # Very wide region as last resort
            (0.30, 0.010, 0.52, 0.035),    # Wide scan across possible positions
        ]

    def extract_region(self, frame: np.ndarray, region: Tuple[float, float, float, float]) -> np.ndarray:
        """Extract a region from the frame using relative coordinates"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = region
        x1, x2 = int(x1 * w), int(x2 * w)
        y1, y2 = int(y1 * h), int(y2 * h)
        return frame[y1:y2, x1:x2]

    def preprocess_for_numbers(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR number detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Scale up for better OCR (EasyOCR works well with larger images)
        scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # Apply slight sharpening to make text clearer
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(scaled, -1, kernel)

        return sharpened

    def parse_stage_round(self, text: str) -> Optional[Tuple[int, int]]:
        """
        Parse stage and round from OCR text

        Expected formats:
        - "1-2" -> (1, 2)
        - "3-4" -> (3, 4)
        - "1 2" -> (1, 2) (in case dash is misread)
        - "1:2" -> (1, 2) (in case dash is misread as colon)

        Returns:
            Tuple of (stage, round) or None if not found
        """
        # Clean the text
        text = text.strip()

        # Pattern 1: "STAGE-ROUND" with dash
        match = re.search(r'(\d+)\s*[-–—]\s*(\d+)', text)
        if match:
            stage = int(match.group(1))
            round_num = int(match.group(2))
            # Validate reasonable ranges
            if 1 <= stage <= 9 and 1 <= round_num <= 7:
                return (stage, round_num)

        # Pattern 2: "STAGE:ROUND" with colon (common OCR error)
        match = re.search(r'(\d+)\s*[:]\s*(\d+)', text)
        if match:
            stage = int(match.group(1))
            round_num = int(match.group(2))
            if 1 <= stage <= 9 and 1 <= round_num <= 7:
                return (stage, round_num)

        # Pattern 3: "STAGE ROUND" with space
        match = re.search(r'(\d+)\s+(\d+)', text)
        if match:
            stage = int(match.group(1))
            round_num = int(match.group(2))
            if 1 <= stage <= 9 and 1 <= round_num <= 7:
                return (stage, round_num)

        # Pattern 4: "STAGEROUND" no separator (try to parse single digits)
        match = re.search(r'(\d)(\d)', text)
        if match and len(text.strip()) <= 3:  # Only for short strings
            stage = int(match.group(1))
            round_num = int(match.group(2))
            if 1 <= stage <= 9 and 1 <= round_num <= 7:
                return (stage, round_num)

        return None

    def detect_stage_round_in_region(self, region: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect stage and round in a specific region"""

        # Preprocess image
        preprocessed = self.preprocess_for_numbers(region)

        try:
            # Use EasyOCR to detect text
            # allowlist parameter restricts to numbers and dash
            results = self.reader.readtext(
                preprocessed,
                allowlist='0123456789-',
                detail=0  # Return only text, not bounding boxes
            )

            # Try to parse each detected text
            for text in results:
                result = self.parse_stage_round(text)
                if result:
                    return result

        except Exception as e:
            pass

        return None

    def detect_stage_round(self, frame: np.ndarray) -> Optional[dict]:
        """
        Detect stage and round from frame

        Args:
            frame: OpenCV image (numpy array)

        Returns:
            Dict with format {"stage": int, "round": int} or None if not detected
        """

        # Check main region first
        main_region = self.extract_region(frame, self.round_region)
        result = self.detect_stage_round_in_region(main_region)

        if result:
            stage, round_num = result
            return {"stage": stage, "round": round_num}

        # Check alternative regions
        for alt_region in self.alt_regions:
            region = self.extract_region(frame, alt_region)
            result = self.detect_stage_round_in_region(region)
            if result:
                stage, round_num = result
                return {"stage": stage, "round": round_num}

        return None

    def get_debug_info(self, frame: np.ndarray) -> dict:
        """Get detailed debug information about round detection"""
        info = {
            'stage_round': None,
            'regions_checked': [],
            'text_found': []
        }

        # Check main region
        main_region = self.extract_region(frame, self.round_region)
        preprocessed = self.preprocess_for_numbers(main_region)

        try:
            results = self.reader.readtext(
                preprocessed,
                allowlist='0123456789-',
                detail=0
            )
            info['text_found'].append(('main', ', '.join(results) if results else 'NO_TEXT'))
        except:
            info['text_found'].append(('main', 'OCR_FAILED'))

        result = self.detect_stage_round_in_region(main_region)
        if result:
            info['stage_round'] = result
            info['regions_checked'].append('main')
            return info

        # Check alternatives
        for i, alt_region in enumerate(self.alt_regions):
            region = self.extract_region(frame, alt_region)
            result = self.detect_stage_round_in_region(region)
            if result:
                info['stage_round'] = result
                info['regions_checked'].append(f'alt_{i}')
                break

        return info
