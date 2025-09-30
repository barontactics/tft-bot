#!/usr/bin/env python3
"""TFT Augment Selection Detector - Final version with strict detection"""

import cv2
import numpy as np

class TFTAugmentDetector:
    """Detects augment selection screens in TFT gameplay"""

    def __init__(self):
        # Text region for "Choose One" (centered at top)
        self.text_region = (0.42, 0.18, 0.58, 0.23)

        # Card regions (3 cards centered in middle of screen)
        self.card_width = 0.18
        self.card_height = 0.42
        self.card_y_center = 0.52
        self.card_x_positions = [0.295, 0.5, 0.705]

        # Purple color range for cards
        self.purple_lower = np.array([130, 30, 30])
        self.purple_upper = np.array([180, 255, 255])

    def detect_transition(self, frame):
        """Check if frame is a transition screen (mostly black)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        black_pixels = np.sum(gray < 10)
        black_ratio = black_pixels / gray.size
        return black_ratio > 0.80

    def is_loading_screen(self, frame):
        """Check if frame is a loading screen (8 champion cards)"""
        h, w = frame.shape[:2]

        # Loading screens have high purple content (8 cards spread across screen)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.purple_lower, self.purple_upper)
        purple_ratio = np.sum(mask > 0) / mask.size

        # Loading screens have >40% purple content (8 cards)
        # Augment screens have 20-35% purple content (3 cards)
        if purple_ratio > 0.40:
            # Additionally check text region
            text_x1 = int(w * self.text_region[0])
            text_x2 = int(w * self.text_region[2])
            text_y1 = int(h * self.text_region[1])
            text_y2 = int(h * self.text_region[3])
            text_area = frame[text_y1:text_y2, text_x1:text_x2]

            text_gray = cv2.cvtColor(text_area, cv2.COLOR_BGR2GRAY)
            _, text_binary = cv2.threshold(text_gray, 200, 255, cv2.THRESH_BINARY)
            white_ratio = np.sum(text_binary > 0) / text_binary.size

            num_labels, _ = cv2.connectedComponents(text_binary)

            # Loading screens: high purple + (no white text OR too many components)
            if white_ratio < 0.10 or num_labels > 20:
                return True

        return False

    def detect_augment(self, frame):
        """
        Detect augment selection screen
        Requirements:
        - NOT a transition screen
        - NOT a loading screen
        - Has "Choose One" text (white text in specific region)
        - Has exactly 3 purple cards in augment positions
        """
        # Quick rejection checks
        if self.detect_transition(frame):
            return False

        if self.is_loading_screen(frame):
            return False

        h, w = frame.shape[:2]

        # Check for "Choose One" text
        text_x1 = int(w * self.text_region[0])
        text_x2 = int(w * self.text_region[2])
        text_y1 = int(h * self.text_region[1])
        text_y2 = int(h * self.text_region[3])
        text_area = frame[text_y1:text_y2, text_x1:text_x2]

        # Check for white text presence
        text_gray = cv2.cvtColor(text_area, cv2.COLOR_BGR2GRAY)
        _, text_binary = cv2.threshold(text_gray, 200, 255, cv2.THRESH_BINARY)
        white_ratio = np.sum(text_binary > 0) / text_binary.size

        # "Choose One" should be 10-30% white pixels
        if not (0.10 <= white_ratio <= 0.30):
            return False

        # Check connected components (letters in "Choose One")
        num_labels, _ = cv2.connectedComponents(text_binary)
        if not (8 <= num_labels <= 15):  # "Choose One" has ~10 characters
            return False

        # Check overall purple content
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.purple_lower, self.purple_upper)
        overall_purple = np.sum(mask > 0) / mask.size

        # Augment screens have 20-35% purple (3 cards)
        if not (0.20 <= overall_purple <= 0.35):
            return False

        # Verify exactly 3 purple cards in augment positions
        purple_cards = 0
        card_height = int(h * self.card_height)
        card_width = int(w * self.card_width)
        card_y_center = int(h * self.card_y_center)

        for x_rel in self.card_x_positions:
            x_center = int(w * x_rel)
            x1 = max(0, x_center - card_width // 2)
            x2 = min(w, x_center + card_width // 2)
            y1 = max(0, card_y_center - card_height // 2)
            y2 = min(h, card_y_center + card_height // 2)

            card_region = frame[y1:y2, x1:x2]
            card_hsv = cv2.cvtColor(card_region, cv2.COLOR_BGR2HSV)
            card_mask = cv2.inRange(card_hsv, self.purple_lower, self.purple_upper)
            purple_ratio = np.sum(card_mask > 0) / card_mask.size

            if purple_ratio > 0.15:  # Card has significant purple
                purple_cards += 1

        # Must have exactly 3 purple cards
        return purple_cards == 3

    def get_augment_cards(self, frame):
        """Extract the 3 augment cards if this is an augment screen"""
        if not self.detect_augment(frame):
            return None

        h, w = frame.shape[:2]
        cards = []

        card_height = int(h * self.card_height)
        card_width = int(w * self.card_width)
        card_y_center = int(h * self.card_y_center)

        for i, x_rel in enumerate(self.card_x_positions):
            x_center = int(w * x_rel)
            x1 = max(0, x_center - card_width // 2)
            x2 = min(w, x_center + card_width // 2)
            y1 = max(0, card_y_center - card_height // 2)
            y2 = min(h, card_y_center + card_height // 2)

            card_region = frame[y1:y2, x1:x2]
            cards.append({
                'position': ['left', 'center', 'right'][i],
                'image': card_region,
                'bounds': (x1, y1, x2, y2)
            })

        return cards

    def visualize_detection(self, frame):
        """Draw detection regions on frame for debugging"""
        vis_frame = frame.copy()
        h, w = frame.shape[:2]

        # Draw text region (green)
        text_x1 = int(w * self.text_region[0])
        text_x2 = int(w * self.text_region[2])
        text_y1 = int(h * self.text_region[1])
        text_y2 = int(h * self.text_region[3])
        cv2.rectangle(vis_frame, (text_x1, text_y1), (text_x2, text_y2), (0, 255, 0), 3)
        cv2.putText(vis_frame, "Choose One Text", (text_x1, text_y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw card regions (blue)
        card_height = int(h * self.card_height)
        card_width = int(w * self.card_width)
        card_y_center = int(h * self.card_y_center)

        for i, x_rel in enumerate(self.card_x_positions):
            x_center = int(w * x_rel)
            x1 = max(0, x_center - card_width // 2)
            x2 = min(w, x_center + card_width // 2)
            y1 = max(0, card_y_center - card_height // 2)
            y2 = min(h, card_y_center + card_height // 2)

            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 100, 0), 3)
            label = ['Left', 'Center', 'Right'][i]
            cv2.putText(vis_frame, f"{label} Card", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)

        return vis_frame


# Example usage
if __name__ == "__main__":
    import os
    import glob

    detector = TFTAugmentDetector()

    # Test on screenshots
    screenshots_dir = "screenshots"
    if os.path.exists(screenshots_dir):
        test_files = glob.glob(os.path.join(screenshots_dir, "*.png"))[:5]

        print("Testing augment detector on sample files:")
        for filepath in test_files:
            frame = cv2.imread(filepath)
            if frame is not None:
                is_augment = detector.detect_augment(frame)
                is_loading = detector.is_loading_screen(frame)
                filename = os.path.basename(filepath)

                if is_augment:
                    print(f"  {filename}: AUGMENT SCREEN")
                elif is_loading:
                    print(f"  {filename}: LOADING SCREEN")
                else:
                    print(f"  {filename}: Other")