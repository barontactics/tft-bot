#!/usr/bin/env python3
"""TFT Gold Amount Detector - Detects and extracts gold amount from gameplay"""

import cv2
import numpy as np
import pytesseract
from typing import Optional, Tuple

class TFTGoldDetector:
    """Detects and extracts gold amount from TFT gameplay screens"""

    def __init__(self):
        # Gold region coordinates (relative to screen size)
        # Gold is displayed in the bottom UI area, TRUE CENTER position
        # Based on analysis, gold appears around 81.5-84.5% from top, 47-53% from left
        self.gold_region = {
            'x1_rel': 0.47,   # 47% from left (true center)
            'x2_rel': 0.53,   # 53% from left (6% width)
            'y1_rel': 0.815,  # 81.5% from top
            'y2_rel': 0.845   # 84.5% from top (3% height)
        }

        # Gold color range in HSV (golden/orange-yellow color)
        # Based on analysis: Hue 12-17, Sat 106-151, Val 174-236
        self.gold_color_lower = np.array([10, 80, 150])  # Lower bound for gold/orange
        self.gold_color_upper = np.array([25, 200, 255])  # Upper bound for gold/orange

    def extract_gold_region(self, frame: np.ndarray) -> np.ndarray:
        """Extract the region where gold amount is displayed"""
        h, w = frame.shape[:2]

        x1 = int(w * self.gold_region['x1_rel'])
        x2 = int(w * self.gold_region['x2_rel'])
        y1 = int(h * self.gold_region['y1_rel'])
        y2 = int(h * self.gold_region['y2_rel'])

        return frame[y1:y2, x1:x2]

    def preprocess_for_ocr(self, region: np.ndarray, mask_icon=True) -> np.ndarray:
        """Preprocess the gold region for better OCR accuracy"""
        # Convert to grayscale first
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Scale up for better OCR (4x for small text)
        scaled = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(scaled)

        # Use adaptive threshold for better results with varying backgrounds
        # Gold text is bright, so we can use a simple brightness threshold
        _, binary = cv2.threshold(enhanced, 140, 255, cv2.THRESH_BINARY)

        # Mask out the gold icon area (left portion of the region)
        # The icon typically occupies the left 25-35% of the region
        if mask_icon:
            h, w = binary.shape
            icon_width = int(w * 0.35)  # Mask left 35% where icon appears
            binary[:, :icon_width] = 0  # Set icon area to black

        # Clean up with morphological operations
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Remove small noise
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 20  # Minimum area for valid text
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                cv2.drawContours(binary, [cnt], -1, 0, -1)

        return binary

    def validate_gold_amount(self, gold: int) -> int:
        """
        Validate and clean up gold amount

        Common issues:
        - Gold icon mistaken for "8", causing 8XX values
        - Values like 840, 850, 861 are likely 40, 50, 61
        """
        # If gold starts with 8 and is > 800, it's likely the icon was read as 8
        if 800 <= gold <= 899:
            # Check if removing the leading 8 gives a reasonable value
            potential_actual = gold - 800
            # Most gold values in TFT are between 0-150
            if 0 <= potential_actual <= 150:
                return potential_actual

        # Values above 900 are suspicious unless it's exactly 999 (max theoretical)
        if 900 <= gold < 999:
            # Try removing leading 9 (could be misread icon)
            potential_actual = gold - 900
            if 0 <= potential_actual <= 99:
                return potential_actual

        # Normal validation for reasonable gold amounts
        if 0 <= gold <= 999:
            return gold

        # For any other unusual values, return as-is but it's likely wrong
        return gold

    def detect_gold(self, frame: np.ndarray) -> Optional[int]:
        """
        Detect and extract gold amount from frame

        Args:
            frame: OpenCV image (numpy array)

        Returns:
            Gold amount as integer, or None if not detected
        """
        try:
            # Extract gold region
            gold_region = self.extract_gold_region(frame)

            # Check if region has gold-colored pixels
            hsv = cv2.cvtColor(gold_region, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.gold_color_lower, self.gold_color_upper)
            gold_pixel_ratio = np.sum(mask > 0) / mask.size

            # If not enough gold pixels, probably not a game screen with gold
            if gold_pixel_ratio < 0.01:  # Less than 1% gold pixels
                return None

            # Preprocess for OCR
            processed = self.preprocess_for_ocr(gold_region)

            # OCR with PSM 7 (single text line) or PSM 8 (single word)
            # Gold is typically just a number (0-100+)
            config = '--psm 8 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(processed, config=config)
            text = text.strip()

            # Try to parse as integer
            if text and text.isdigit():
                gold = int(text)
                # Validate and clean the gold amount
                gold = self.validate_gold_amount(gold)
                if 0 <= gold <= 999:
                    return gold

            # If PSM 8 didn't work, try PSM 7
            config = '--psm 7 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(processed, config=config)
            text = text.strip()

            if text and text.isdigit():
                gold = int(text)
                # Validate and clean the gold amount
                gold = self.validate_gold_amount(gold)
                if 0 <= gold <= 999:
                    return gold

        except Exception as e:
            # Silently handle errors (OCR might fail on non-game screens)
            pass

        return None

    def detect_gold_with_confidence(self, frame: np.ndarray) -> Tuple[Optional[int], float]:
        """
        Detect gold with confidence score

        Args:
            frame: OpenCV image (numpy array)

        Returns:
            Tuple of (gold_amount, confidence_score)
            confidence_score is between 0.0 and 1.0
        """
        try:
            gold_region = self.extract_gold_region(frame)

            # Check gold pixel ratio for confidence
            hsv = cv2.cvtColor(gold_region, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.gold_color_lower, self.gold_color_upper)
            gold_pixel_ratio = np.sum(mask > 0) / mask.size

            # Base confidence from gold pixel presence
            pixel_confidence = 0.0
            if gold_pixel_ratio > 0.01:  # At least 1% gold pixels
                # Typical good detection has 5-20% gold pixels
                pixel_confidence = min(1.0, gold_pixel_ratio / 0.15) * 0.4  # 40% weight

            # Try OCR detection
            processed = self.preprocess_for_ocr(gold_region)

            # Try both PSM modes and get OCR confidence
            ocr_results = []

            # PSM 8 - Single word
            config = '--psm 8 -c tessedit_char_whitelist=0123456789'
            try:
                data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
                for i, conf in enumerate(data['conf']):
                    if int(conf) > 0 and data['text'][i].strip().isdigit():
                        gold_val = int(data['text'][i].strip())
                        # Validate and clean the gold amount
                        gold_val = self.validate_gold_amount(gold_val)
                        if 0 <= gold_val <= 999:
                            ocr_results.append((gold_val, int(conf) / 100.0))
            except:
                pass

            # PSM 7 - Single line
            config = '--psm 7 -c tessedit_char_whitelist=0123456789'
            try:
                data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
                for i, conf in enumerate(data['conf']):
                    if int(conf) > 0 and data['text'][i].strip().isdigit():
                        gold_val = int(data['text'][i].strip())
                        # Validate and clean the gold amount
                        gold_val = self.validate_gold_amount(gold_val)
                        if 0 <= gold_val <= 999:
                            ocr_results.append((gold_val, int(conf) / 100.0))
            except:
                pass

            if ocr_results:
                # Take the result with highest OCR confidence
                best_result = max(ocr_results, key=lambda x: x[1])
                gold_amount, ocr_conf = best_result

                # OCR confidence contributes 60% of final confidence
                ocr_confidence = ocr_conf * 0.6

                # Calculate final confidence
                total_confidence = pixel_confidence + ocr_confidence

                # Boost confidence if gold amount is reasonable for game state
                if 50 <= gold_amount <= 150:  # Common gold range
                    total_confidence = min(1.0, total_confidence * 1.1)

                return gold_amount, min(1.0, total_confidence)

            # If no OCR results but gold pixels detected
            elif pixel_confidence > 0:
                # Try simple detect_gold as fallback
                gold = self.detect_gold(frame)
                if gold is not None:
                    # Lower confidence since OCR confidence data wasn't available
                    return gold, min(0.5, pixel_confidence * 1.5)

        except Exception as e:
            pass

        return None, 0.0

    def visualize_gold_region(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw gold detection region on frame for debugging

        Args:
            frame: OpenCV image (numpy array)

        Returns:
            Frame with gold region highlighted
        """
        vis_frame = frame.copy()
        h, w = frame.shape[:2]

        x1 = int(w * self.gold_region['x1_rel'])
        x2 = int(w * self.gold_region['x2_rel'])
        y1 = int(h * self.gold_region['y1_rel'])
        y2 = int(h * self.gold_region['y2_rel'])

        # Draw rectangle around gold region
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 215, 0), 3)  # Gold color

        # Detect gold and add text
        gold = self.detect_gold(frame)
        if gold is not None:
            cv2.putText(vis_frame, f"Gold: {gold}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 215, 0), 2)
        else:
            cv2.putText(vis_frame, "Gold: Not detected", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return vis_frame

    def save_debug_images(self, frame: np.ndarray, output_prefix: str = "gold_debug"):
        """Save debug images for gold detection"""
        gold_region = self.extract_gold_region(frame)
        processed = self.preprocess_for_ocr(gold_region)

        cv2.imwrite(f"{output_prefix}_region.png", gold_region)
        cv2.imwrite(f"{output_prefix}_processed.png", processed)

        vis_frame = self.visualize_gold_region(frame)
        cv2.imwrite(f"{output_prefix}_visualization.png", vis_frame)


# Example usage and testing
if __name__ == "__main__":
    import os
    import glob

    detector = TFTGoldDetector()

    # Test on screenshots
    screenshots_dir = "screenshots"
    if os.path.exists(screenshots_dir):
        # Test on a few round screenshots
        test_files = [
            "screenshots/round_1-3_frame03960_time01m06s.png",
            "screenshots/round_2-3_frame18720_time05m12s.png"
        ]

        for filepath in test_files:
            if os.path.exists(filepath):
                frame = cv2.imread(filepath)
                if frame is not None:
                    gold = detector.detect_gold(frame)
                    gold_conf, confidence = detector.detect_gold_with_confidence(frame)

                    filename = os.path.basename(filepath)
                    print(f"{filename}:")
                    print(f"  Gold: {gold}")
                    print(f"  Confidence: {confidence:.2f}")

                    # Save debug images for first file
                    if filepath == test_files[0]:
                        detector.save_debug_images(frame, "gold_debug")