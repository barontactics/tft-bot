#!/usr/bin/env python3
"""TFT Level Detector - Detects and extracts current Level from gameplay"""

import cv2
import numpy as np
import pytesseract
from typing import Optional, Tuple

# Try to import EasyOCR (optional)
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

class TFTLevelDetector:
    """Detects and extracts the level from TFT gameplay screens"""
    pass

    def __init__(self, use_easyocr=True):
        # Level region coordinates (relative to screen size)
        # Level is displayed in the bottom UI area, TK position
        # ???? Optimized: 5% width with 35% spatial masking - best balance

        self.level_region = {
                'x1_rel': 0.1385,  # 13.85% from left
                'x2_rel': 0.1885,  # 18.85% from left (9% width)
                'y1_rel': 0.815,  # 81.5% from top
                'y2_rel': 0.845   # 84.5% from top (3% height)
            }
    
        # Initialize EasyOCR reader if available and requested
        self.easyocr_reader = None
        if use_easyocr and EASYOCR_AVAILABLE:
            try:
                # Initialize with English only for faster performance
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                print("EasyOCR initialized successfully")
            except Exception as e:
                print(f"Failed to initialize EasyOCR: {e}")
                self.easyocr_reader = None

    def extract_level_region(self, frame: np.ndarray) -> np.ndarray:
        """Extract the region where level is displayed"""
        h, w = frame.shape[:2]

        x1 = int(w * self.level_region['x1_rel'])
        x2 = int(w * self.level_region['x2_rel'])
        y1 = int(h * self.level_region['y1_rel'])
        y2 = int(h * self.level_region['y2_rel'])

        return frame[y1:y2, x1:x2]
    
    def preprocess_for_ocr(self, region: np.ndarray) -> np.ndarray:
        """Preprocess the level region for OCR"""
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Scale up for better OCR
        scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(scaled)

        # Threshold to get white text on dark background
        # Level numbers are typically white/bright
        _, binary = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)

        # Clean up with morphological operations
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

    def detect_level_easyocr(self, frame: np.ndarray) -> Optional[int]:
        """
        Detect level using EasyOCR

        Args:
            frame: OpenCV image (numpy array)

        Returns:
            Level as integer, or None if not detected
        """
        if self.easyocr_reader is None:
            return None

        try:
            # Extract level region
            level_region = self.extract_level_region(frame)

            # Preprocess for OCR
            processed = self.preprocess_for_ocr(level_region)

            # EasyOCR works better with RGB
            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            print("Processed rgb")

            # Run EasyOCR with allowlist for digits only
            results = self.easyocr_reader.readtext(
                processed_rgb,
                allowlist='0123456789',
                detail=1  # Return confidence scores
            )
            print(f"EasyOCR results: {results}")

            # Filter results
            for bbox, text, confidence in results:
                text = text.strip()
                if text and text.isdigit():
                    level = int(text)
                    if 0 <= level <= 10:
                        return level

        except Exception as e:
            print(f"Error in detect_level_easyocr: {e}")
            import traceback
            traceback.print_exc()

        return None
    def detect_level(self, frame: np.ndarray) -> Optional[int]:
        """
        Detect and extract level from frame using ensemble approach

        Args:
            frame: OpenCV image (numpy array)

        Returns:
            Level as integer, or None if not detected
        """
        # Try EasyOCR first if available
        if self.easyocr_reader is not None:
            print("Trying EasyOCR for level detection")
            easyocr_result = self.detect_level_easyocr(frame)
            if easyocr_result is not None:
                return easyocr_result

        # Fallback to Tesseract
        try:
            # Extract level region
            level_region = self.extract_level_region(frame)

            # Preprocess for OCR
            processed = self.preprocess_for_ocr(level_region)

            # OCR with PSM 7 (single text line) or PSM 8 (single word)
    
            config = '--psm 8 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(processed, config=config)
            text = text.strip()

            # Try to parse 
            if text and text.isdigit():
                level = int(text)
                if 0 <= level <= 10:
                    return level

            # If PSM 8 didn't work, try PSM 7
            config = '--psm 7 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(processed, config=config)
            text = text.strip()

            if text and text.isdigit():
                level = int(text)
                if 0 <= level <= 10:
                    return level

        except Exception as e:
            # Silently handle errors (OCR might fail on non-game screens)
            print(f"Error in detect_level: {e}")
            import traceback
            traceback.print_exc()

        return None

    def visualize_level_region(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw level detection region on frame for debugging

        Args:
            frame: OpenCV image (numpy array)

        Returns:
            Frame with level region highlighted
        """
        vis_frame = frame.copy()
        h, w = frame.shape[:2]

        x1 = int(w * self.level_region['x1_rel'])
        x2 = int(w * self.level_region['x2_rel'])
        y1 = int(h * self.level_region['y1_rel'])
        y2 = int(h * self.level_region['y2_rel'])

        # Draw rectangle around level region
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 215, 0), 3)  # Gold color

        #Detect level and add text
        level = self.detect_level(frame)
        if level is not None:
            cv2.putText(vis_frame, f"Level: {level}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 215, 0), 2)
        else:
            cv2.putText(vis_frame, "Level: Not detected", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return vis_frame

    def save_debug_images(self, frame: np.ndarray, output_prefix: str = "level_debug"):
        """Save debug images for level detection"""
        level_region = self.extract_level_region(frame)
        processed = self.preprocess_for_ocr(level_region)

        cv2.imwrite(f"{output_prefix}_region.png", level_region)
        cv2.imwrite(f"{output_prefix}_processed.png", processed)

        vis_frame = self.visualize_level_region(frame)
        cv2.imwrite(f"{output_prefix}_visualization.png", vis_frame)


# Example usage and testing
if __name__ == "__main__":
    import os
    import glob

    detector = TFTLevelDetector()

    # Test on snapshots from gold detector tests
    snapshots_base = "tft_video_analyzer/detectors/levels/tests/snapshots"
    test_files = [
        os.path.join(snapshots_base, "snapshot_000", "frame.png"),
        os.path.join(snapshots_base, "snapshot_010", "frame.png"),
        os.path.join(snapshots_base, "snapshot_004", "frame.png"),
        #os.path.join(snapshots_base, "snapshot_021", "frame.png"),
        os.path.join(snapshots_base, "snapshot_026", "frame.png"),
        #os.path.join(snapshots_base, "snapshot_029", "frame.png"),
    ]

    for filepath in test_files:
        if os.path.exists(filepath):
            frame = cv2.imread(filepath)
            if frame is not None:
                # Extract snapshot number from path (e.g., snapshot_000)
                snapshot_dir = os.path.basename(os.path.dirname(filepath))
                print(f"\nTesting {snapshot_dir}:")
                # Detect level
                level_string = detector.detect_level(frame)
                print(f"  Detected Level: {level_string if level_string else 'Not detected'}")
                
                # Extract and display level region info
                level_region = detector.extract_level_region(frame)
                print(f"  Level region shape: {level_region.shape}")
                print(f"  Level region size: {level_region.shape[1]}x{level_region.shape[0]} pixels")

                # Save debug images with unique snapshot-based names
                output_prefix = f"level_debug_{snapshot_dir}"
                detector.save_debug_images(frame, output_prefix)
                print(f"  Saved debug images with prefix: {output_prefix}")
                print(f"    - {output_prefix}_region.png (extracted region)")
                print(f"    - {output_prefix}_visualization.png (highlighted on frame)")
        else:
            print(f"File not found: {filepath}")

