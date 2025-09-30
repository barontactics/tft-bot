#!/usr/bin/env python3
"""TFT Streak Detector - Detects win/loss streaks and their length"""

import cv2
import numpy as np
import pytesseract
from typing import Optional, Tuple, Dict
from enum import Enum

class StreakType(Enum):
    """Enumeration for streak types"""
    WIN = "win"
    LOSS = "loss"
    NONE = "none"

class TFTStreakDetector:
    """Detects and extracts streak type and length from TFT gameplay screens"""

    def __init__(self):
        # Streak region coordinates (relative to screen size)
        # Streak is displayed to the right of gold region
        # Smaller region focusing on the darker area where number appears
        self.streak_region = {
            'x1_rel': 0.5425, # 54.25% from left (moved 1% right)
            'x2_rel': 0.5825, # 58.25% from left (4% width)
            'y1_rel': 0.810,  # 81% from top
            'y2_rel': 0.840   # 84% from top (3% height)
        }

        # Color ranges in HSV for streak types
        # Blue fire for loss streak
        self.blue_lower = np.array([100, 50, 50])
        self.blue_upper = np.array([130, 255, 255])

        # Red/orange fire for win streak
        self.red_lower1 = np.array([0, 50, 50])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 50, 50])
        self.red_upper2 = np.array([180, 255, 255])

    def extract_streak_region(self, frame: np.ndarray) -> np.ndarray:
        """Extract the region where streak is displayed"""
        h, w = frame.shape[:2]

        x1 = int(w * self.streak_region['x1_rel'])
        x2 = int(w * self.streak_region['x2_rel'])
        y1 = int(h * self.streak_region['y1_rel'])
        y2 = int(h * self.streak_region['y2_rel'])

        return frame[y1:y2, x1:x2]

    def detect_streak_type(self, region: np.ndarray) -> StreakType:
        """
        Detect if streak is win, loss, or none based on flame color

        Args:
            region: The streak region image

        Returns:
            StreakType enum value
        """
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Check for blue fire (loss streak)
        blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        blue_ratio = np.sum(blue_mask > 0) / blue_mask.size

        # Check for red/orange fire (win streak)
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_ratio = np.sum(red_mask > 0) / red_mask.size

        # Determine streak type based on color ratios
        # Need at least 2% of pixels to be colored for valid detection
        if blue_ratio > 0.02 and blue_ratio > red_ratio:
            return StreakType.LOSS
        elif red_ratio > 0.02 and red_ratio > blue_ratio:
            return StreakType.WIN
        else:
            return StreakType.NONE

    def preprocess_for_ocr(self, region: np.ndarray) -> np.ndarray:
        """Preprocess the streak region for OCR to extract number"""
        # Focus on the right 60% where the number typically appears
        h, w = region.shape[:2]
        number_region = region[:, int(w*0.4):]

        # Convert to grayscale
        gray = cv2.cvtColor(number_region, cv2.COLOR_BGR2GRAY)

        # Scale up for better OCR (6x for small text)
        scaled = cv2.resize(gray, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(scaled)

        # Use higher threshold to avoid merging digits
        _, binary = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY)

        # Clean up with smaller morphological operations to preserve digit separation
        kernel = np.ones((1, 1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Remove small noise
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 15  # Minimum area for valid text
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                cv2.drawContours(binary, [cnt], -1, 0, -1)

        return binary

    def extract_streak_length(self, region: np.ndarray) -> Optional[int]:
        """
        Extract the streak length number using OCR

        Args:
            region: The streak region image

        Returns:
            Streak length as integer, or None if not detected
        """
        try:
            # Preprocess for OCR
            processed = self.preprocess_for_ocr(region)

            # Try different OCR configurations
            configs = [
                '--psm 8 -c tessedit_char_whitelist=0123456789',  # Single word
                '--psm 7 -c tessedit_char_whitelist=0123456789',  # Single line
                '--psm 13 -c tessedit_char_whitelist=0123456789', # Raw line
                '-c tessedit_char_whitelist=0123456789',          # Default
            ]

            for config in configs:
                text = pytesseract.image_to_string(processed, config=config).strip()

                if text and text.isdigit():
                    length = int(text)
                    # Validate reasonable streak length (0-15)
                    if 0 <= length <= 15:
                        return length

        except Exception:
            pass

        return None

    def detect_streak(self, frame: np.ndarray) -> Dict:
        """
        Detect streak type and length from frame

        Args:
            frame: OpenCV image (numpy array)

        Returns:
            Dictionary with keys:
                - 'type': StreakType enum (WIN, LOSS, or NONE)
                - 'length': Integer streak length or None (or -1 if no streak detected)
                - 'confidence': Float confidence score (0.0 to 1.0)
        """
        # Extract streak region
        streak_region = self.extract_streak_region(frame)

        # Detect streak type
        streak_type = self.detect_streak_type(streak_region)

        # Extract streak length
        streak_length = self.extract_streak_length(streak_region)

        # Calculate confidence based on type detection and OCR success
        confidence = 0.0

        # If no streak detected (NONE type), return empty type and -1 length
        if streak_type == StreakType.NONE:
            return {
                'type': StreakType.NONE,  # Will be converted to empty string in analysis script
                'length': -1,
                'confidence': 0.0
            }

        # Streak detected, calculate confidence
        confidence = 0.5  # Base confidence for detecting streak type

        if streak_length is not None:
            confidence += 0.5  # Additional confidence for successful OCR

            # If length is 0 but we detect a streak type, it's likely correct
            if streak_length == 0:
                confidence = 0.9
        else:
            # If we detect a streak but can't read the length, still return -1
            streak_length = -1
            confidence = 0.5  # Lower confidence without length

        return {
            'type': streak_type,
            'length': streak_length,
            'confidence': confidence
        }

    def visualize_streak_region(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw streak detection region on frame for debugging

        Args:
            frame: OpenCV image (numpy array)

        Returns:
            Frame with streak region highlighted
        """
        vis_frame = frame.copy()
        h, w = frame.shape[:2]

        # Draw streak region
        x1 = int(w * self.streak_region['x1_rel'])
        x2 = int(w * self.streak_region['x2_rel'])
        y1 = int(h * self.streak_region['y1_rel'])
        y2 = int(h * self.streak_region['y2_rel'])

        # Detect streak
        result = self.detect_streak(frame)
        streak_type = result['type']
        streak_length = result['length']

        # Choose color based on streak type
        if streak_type == StreakType.WIN:
            color = (0, 0, 255)  # Red for win
            label = f"Win Streak: {streak_length if streak_length is not None else '?'}"
        elif streak_type == StreakType.LOSS:
            color = (255, 0, 0)  # Blue for loss
            label = f"Loss Streak: {streak_length if streak_length is not None else '?'}"
        else:
            color = (128, 128, 128)  # Gray for none
            label = "No Streak"

        # Draw rectangle
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)

        # Add label
        cv2.putText(vis_frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Also draw gold region for reference
        gold_x1 = int(w * 0.47)
        gold_x2 = int(w * 0.53)
        cv2.rectangle(vis_frame, (gold_x1, y1), (gold_x2, y2), (255, 215, 0), 2)
        cv2.putText(vis_frame, "Gold", (gold_x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 215, 0), 2)

        return vis_frame

    def save_debug_images(self, frame: np.ndarray, output_prefix: str = "streak_debug"):
        """Save debug images for streak detection"""
        streak_region = self.extract_streak_region(frame)
        processed = self.preprocess_for_ocr(streak_region)

        cv2.imwrite(f"{output_prefix}_region.png", streak_region)
        cv2.imwrite(f"{output_prefix}_processed.png", processed)

        # Create color masks
        hsv = cv2.cvtColor(streak_region, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        cv2.imwrite(f"{output_prefix}_blue_mask.png", blue_mask)
        cv2.imwrite(f"{output_prefix}_red_mask.png", red_mask)

        vis_frame = self.visualize_streak_region(frame)
        cv2.imwrite(f"{output_prefix}_visualization.png", vis_frame)


# Example usage and testing
if __name__ == "__main__":
    import os
    import glob

    detector = TFTStreakDetector()

    # Test on screenshots
    test_files = [
        "screenshots/round_1-3_frame03960_time01m06s.png",
        "screenshots/round_2-3_frame18720_time05m12s.png",
        "screenshots/round_3-5_frame53820_time14m57s.png",
    ]

    print("Testing Streak Detection")
    print("=" * 60)

    for filepath in test_files:
        if os.path.exists(filepath):
            frame = cv2.imread(filepath)
            if frame is not None:
                result = detector.detect_streak(frame)

                filename = os.path.basename(filepath)
                print(f"\n{filename}:")
                print(f"  Type: {result['type'].value}")
                print(f"  Length: {result['length']}")
                print(f"  Confidence: {result['confidence']:.2f}")

                # Save debug images for first file
                if filepath == test_files[0]:
                    detector.save_debug_images(frame, "streak_debug")

    print("\nDebug images saved: streak_debug_*.png")