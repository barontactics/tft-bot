#!/usr/bin/env python3
"""TFT Health Detector - Detects all players' health from the right sidebar"""

import cv2
import numpy as np
import pytesseract
from typing import Optional, Dict, Tuple, List

class TFTHealthDetector:
    """Detects and extracts all players' health from TFT gameplay screens"""

    def __init__(self):
        # Right sidebar region where player list appears
        # Players are listed vertically on the right side
        self.sidebar_region = {
            'x1_rel': 0.85,   # 85% from left (right side of screen)
            'x2_rel': 0.97,   # 97% from left (12% width)
            'y1_rel': 0.16,   # 16% from top (skip top UI)
            'y2_rel': 0.71    # 71% from top (55% height total)
        }

        # Yellow color range for identifying user's player (in HSV)
        # Yellow circle around the user's icon
        self.yellow_lower = np.array([20, 100, 100])
        self.yellow_upper = np.array([35, 255, 255])

        # Each player slot height is exactly 7.3% of screen
        self.player_slot_height_rel = 0.073

        # Maximum number of players in TFT
        self.max_players = 8

        # Health bar region within each slot (left portion)
        self.health_region_width_rel = 0.35  # 35% of sidebar width for health number

    def extract_sidebar_region(self, frame: np.ndarray) -> np.ndarray:
        """Extract the right sidebar region containing player list"""
        h, w = frame.shape[:2]

        x1 = int(w * self.sidebar_region['x1_rel'])
        x2 = int(w * self.sidebar_region['x2_rel'])
        y1 = int(h * self.sidebar_region['y1_rel'])
        y2 = int(h * self.sidebar_region['y2_rel'])

        return frame[y1:y2, x1:x2]

    def find_user_player_position(self, sidebar: np.ndarray) -> Optional[int]:
        """
        Find the vertical position of the user's player by detecting yellow circle

        Returns:
            Y-coordinate of the user's player slot center, or None if not found
        """
        # Convert to HSV for yellow detection
        hsv = cv2.cvtColor(sidebar, cv2.COLOR_BGR2HSV)

        # Create mask for yellow color
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)

        # Find contours of yellow regions
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find the largest yellow contour (should be the circle)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Check if it's circular enough (aspect ratio close to 1)
        if 0.7 < w/h < 1.3 and cv2.contourArea(largest_contour) > 100:
            # Return center Y position
            return y + h // 2

        return None

    def extract_health_region(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Extract the health region for the user's player

        Returns:
            Tuple of (health_region, info_dict) or None if not found
        """
        sidebar = self.extract_sidebar_region(frame)

        # Find user's player position
        user_y = self.find_user_player_position(sidebar)

        if user_y is None:
            return None

        # Calculate health region based on player position
        # Health number is typically to the left of the icon
        h, w = sidebar.shape[:2]
        frame_h, frame_w = frame.shape[:2]

        # Health text is in the black bar area, left side of the player slot
        slot_height = int(frame_h * self.player_slot_height_rel)

        # Extract region around the detected player
        y1 = max(0, user_y - slot_height // 2)
        y2 = min(h, user_y + slot_height // 2)

        # Health number is in the left portion of the slot
        health_region = sidebar[y1:y2, :int(w * 0.4)]

        # Calculate absolute position in frame
        abs_x1 = int(frame_w * self.sidebar_region['x1_rel'])
        abs_y1 = int(frame_h * self.sidebar_region['y1_rel']) + y1
        abs_x2 = abs_x1 + int(w * 0.4)
        abs_y2 = int(frame_h * self.sidebar_region['y1_rel']) + y2

        info = {
            'sidebar_y': user_y,
            'abs_coords': (abs_x1, abs_y1, abs_x2, abs_y2)
        }

        return health_region, info

    def preprocess_for_ocr(self, region: np.ndarray) -> np.ndarray:
        """Preprocess the health region for OCR"""
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Scale up for better OCR
        scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(scaled)

        # Threshold to get white text on dark background
        # Health numbers are typically white/bright
        _, binary = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)

        # Clean up with morphological operations
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

    def extract_player_slots(self, frame: np.ndarray) -> List[Tuple[np.ndarray, int]]:
        """
        Extract individual player slots from the sidebar

        Returns:
            List of tuples (slot_image, slot_index) for up to 8 players
        """
        slots = []
        sidebar = self.extract_sidebar_region(frame)
        sidebar_h, sidebar_w = sidebar.shape[:2]
        frame_h = frame.shape[0]

        # Each slot is 7.3% of the frame height
        slot_height = int(frame_h * self.player_slot_height_rel)

        # Calculate slot height within the sidebar
        # Since sidebar is 55% of frame height, we need to scale appropriately
        slot_height_in_sidebar = int(slot_height * (sidebar_h / (frame_h * 0.55)))

        for i in range(self.max_players):
            # Calculate position in sidebar
            y1 = i * slot_height_in_sidebar
            y2 = min((i + 1) * slot_height_in_sidebar, sidebar_h)

            if y1 >= sidebar_h:
                break

            slot = sidebar[y1:y2, :]

            # Check if slot has content (not pure black)
            if np.mean(slot) > 10:  # Not empty/black
                slots.append((slot, i))

        return slots

    def extract_health_and_name_from_slot(self, slot: np.ndarray) -> Tuple[Optional[int], Optional[str]]:
        """
        Extract both health value and player name from a single player slot using full-slot OCR

        Args:
            slot: Image of a single player slot

        Returns:
            Tuple of (health_value, player_name) or (None, None) if not detected
        """
        # Convert to grayscale
        gray = cv2.cvtColor(slot, cv2.COLOR_BGR2GRAY)

        # Scale up for better OCR
        scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # Apply simple threshold (works best based on testing)
        _, binary = cv2.threshold(scaled, 150, 255, cv2.THRESH_BINARY)

        # Try PSM 11 (sparse text) which works best for game UI
        try:
            # First try with alphanumeric whitelist for cleaner results
            config_alnum = '--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 '
            text = pytesseract.image_to_string(binary, config=config_alnum).strip()

            if not text:
                # Fallback to PSM 3 if PSM 11 doesn't work
                config_fallback = '--psm 3'
                text = pytesseract.image_to_string(binary, config=config_fallback).strip()

            # Parse the OCR result
            if text:
                # Clean up and split the text
                lines = [line.strip() for line in text.split('\n') if line.strip()]

                health = None
                name = None

                for line in lines:
                    # Split by common separators
                    parts = line.replace('|', ' ').split()

                    for part in parts:
                        # Check if it's a health value (1-100)
                        if part.isdigit():
                            val = int(part)
                            if 1 <= val <= 100:
                                health = val
                        # Check if it's mixed alphanumeric (like "mDBepo92")
                        elif any(c.isdigit() for c in part) and any(c.isalpha() for c in part):
                            # Try to extract numbers from the end
                            import re
                            # Match pattern to separate name from trailing numbers
                            # This captures everything up to the last sequence of digits
                            match = re.match(r'(.+?)(\d{1,3})$', part)
                            if match:
                                potential_name = match.group(1)
                                potential_health = match.group(2)

                                # Validate health value
                                if potential_health.isdigit():
                                    val = int(potential_health)
                                    if 1 <= val <= 100:
                                        health = val
                                        # Also extract the name part
                                        if len(potential_name) > 2:
                                            name = potential_name
                        # Otherwise it might be part of the name
                        elif len(part) > 2 and part.isalpha():
                            if name is None:
                                name = part
                            elif part.lower() not in ['x', 'f', 've']:  # Filter common OCR artifacts
                                name = part if len(part) > len(name) else name

                return health, name

        except Exception:
            pass

        # If full slot OCR fails, try the original methods as fallback
        return self.extract_health_from_slot(slot), self.extract_name_from_slot(slot)

    def extract_health_from_slot(self, slot: np.ndarray) -> Optional[int]:
        """
        Fallback method to extract just health value from a single player slot

        Args:
            slot: Image of a single player slot

        Returns:
            Health value or None if not detected
        """
        h, w = slot.shape[:2]

        # Health is displayed in a black bar that starts from the left side
        health_region = slot[:, int(w*0.30):int(w*0.50)]

        # Convert to grayscale
        gray = cv2.cvtColor(health_region, cv2.COLOR_BGR2GRAY)

        # Scale up significantly for better OCR
        scaled = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(scaled)

        # Threshold to extract white text on dark background
        _, binary = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY)

        # Clean up with small kernel
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Try OCR with different configurations
        configs = [
            '--psm 7 -c tessedit_char_whitelist=0123456789',  # Single line
            '--psm 8 -c tessedit_char_whitelist=0123456789',  # Single word
            '--psm 13 -c tessedit_char_whitelist=0123456789', # Raw line
        ]

        for config in configs:
            try:
                text = pytesseract.image_to_string(binary, config=config).strip()

                if text and text.isdigit():
                    health = int(text)
                    if 1 <= health <= 100:
                        return health
            except Exception:
                pass

        return None

    def extract_name_from_slot(self, slot: np.ndarray) -> Optional[str]:
        """
        Extract player name from a single player slot

        Args:
            slot: Image of a single player slot

        Returns:
            Player name or None if not detected
        """
        h, w = slot.shape[:2]

        # Name is in the middle portion of the slot
        name_region = slot[:h//2, int(w * 0.35):int(w * 0.75)]

        # Convert to grayscale and threshold
        gray = cv2.cvtColor(name_region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Try OCR
        try:
            text = pytesseract.image_to_string(binary, config='--psm 8').strip()

            # Clean up the text
            text = ''.join(c for c in text if c.isalnum() or c in ' _-')

            if len(text) > 2:  # Reasonable name length
                return text
        except Exception:
            pass

        return None

    def is_user_slot(self, slot: np.ndarray) -> bool:
        """
        Check if this slot belongs to the user (has yellow circle)

        Args:
            slot: Image of a single player slot

        Returns:
            True if this is the user's slot
        """
        # Check for yellow circle in the right portion (where icon is)
        h, w = slot.shape[:2]
        icon_region = slot[:, int(w * 0.65):]

        # Convert to HSV and check for yellow
        hsv = cv2.cvtColor(icon_region, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)

        # Check if enough yellow pixels
        yellow_ratio = np.sum(yellow_mask > 0) / yellow_mask.size

        # Set threshold at 6% to catch true yellow circles (typically 8%+)
        # but may include some false positives from golden UI elements
        # Slot 1 (user) = 8.2%, Slot 6 (not user but golden) = 9.2%
        # This is tricky as slot 6 has more yellow than the actual user slot
        return yellow_ratio > 0.06 and yellow_ratio < 0.09  # Between 6% and 9% yellow pixels

    def detect_all_players_health(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect health of all players in the game

        Returns:
            List of dictionaries, each containing:
                - 'slot_index': Index of the player slot (0-7)
                - 'health': Integer health value or None
                - 'name': Player name or None
                - 'is_user': Boolean indicating if this is the user's player
                - 'position': Absolute coordinates in frame
        """
        slots = self.extract_player_slots(frame)

        results = []
        frame_h, frame_w = frame.shape[:2]
        sidebar_x = int(frame_w * self.sidebar_region['x1_rel'])
        sidebar_y = int(frame_h * self.sidebar_region['y1_rel'])

        for slot, slot_idx in slots:
            if slot_idx >= self.max_players:
                break

            # Check if this is the user's slot first
            is_user = self.is_user_slot(slot)

            # Extract information from slot using full-slot OCR
            health, name = self.extract_health_and_name_from_slot(slot)

            # Skip name for user slots (they don't display names)
            if is_user:
                name = None

            # Calculate absolute position in frame
            # Each slot is 7.3% of the frame height
            slot_height = int(frame_h * self.player_slot_height_rel)
            abs_y1 = sidebar_y + slot_idx * slot_height
            abs_y2 = abs_y1 + slot_height
            abs_x1 = sidebar_x
            abs_x2 = int(frame_w * self.sidebar_region['x2_rel'])

            results.append({
                'slot_index': slot_idx,
                'health': health,
                'name': name,
                'is_user': is_user,
                'position': (abs_x1, abs_y1, abs_x2, abs_y2)
            })

        return results

    def extract_health_value(self, region: np.ndarray) -> Optional[int]:
        """
        Extract the health value using OCR

        Returns:
            Health value as integer, or None if not detected
        """
        try:
            # Preprocess for OCR
            processed = self.preprocess_for_ocr(region)

            # Try different OCR configurations
            configs = [
                '--psm 8 -c tessedit_char_whitelist=0123456789',  # Single word
                '--psm 7 -c tessedit_char_whitelist=0123456789',  # Single line
                '--psm 11 -c tessedit_char_whitelist=0123456789', # Sparse text
            ]

            for config in configs:
                text = pytesseract.image_to_string(processed, config=config).strip()

                if text and text.isdigit():
                    health = int(text)
                    # Validate reasonable health range (1-100 in TFT)
                    if 1 <= health <= 100:
                        return health

        except Exception:
            pass

        return None

    def detect_health(self, frame: np.ndarray) -> Dict:
        """
        Detect user's player health from frame

        Returns:
            Dictionary with keys:
                - 'health': Integer health value or None
                - 'found': Boolean indicating if user's player was found
                - 'confidence': Float confidence score (0.0 to 1.0)
        """
        result = self.extract_health_region(frame)

        if result is None:
            return {
                'health': None,
                'found': False,
                'confidence': 0.0
            }

        health_region, info = result

        # Extract health value
        health = self.extract_health_value(health_region)

        # Calculate confidence
        confidence = 0.5 if health is not None else 0.3  # Base confidence for finding player
        if health is not None:
            confidence = 0.9  # High confidence if OCR succeeded

        return {
            'health': health,
            'found': True,
            'confidence': confidence,
            'position': info
        }

    def visualize_all_players(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw all players' health detection visualization on frame

        Returns:
            Frame with all player slots and health values highlighted
        """
        vis_frame = frame.copy()

        # Detect all players
        players = self.detect_all_players_health(frame)

        for player in players:
            x1, y1, x2, y2 = player['position']

            # Choose color based on detection status and user
            if player['is_user']:
                color = (0, 255, 255)  # Yellow for user
            elif player['health'] is not None:
                color = (0, 255, 0)  # Green for successful detection
            else:
                color = (0, 0, 255)  # Red for failed detection

            # Draw rectangle around player slot
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

            # Add label with player info
            label_parts = []
            if player['name']:
                label_parts.append(player['name'])
            if player['health'] is not None:
                label_parts.append(f"HP: {player['health']}")
            else:
                label_parts.append("HP: ?")
            if player['is_user']:
                label_parts.append("(YOU)")

            label = " ".join(label_parts)
            cv2.putText(vis_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw the sidebar region for reference
        h, w = frame.shape[:2]
        sidebar_x1 = int(w * self.sidebar_region['x1_rel'])
        sidebar_x2 = int(w * self.sidebar_region['x2_rel'])
        sidebar_y1 = int(h * self.sidebar_region['y1_rel'])
        sidebar_y2 = int(h * self.sidebar_region['y2_rel'])

        cv2.rectangle(vis_frame, (sidebar_x1, sidebar_y1), (sidebar_x2, sidebar_y2),
                     (128, 128, 128), 1)
        cv2.putText(vis_frame, f"Players: {len(players)}/8", (sidebar_x1, sidebar_y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return vis_frame

    def visualize_health_detection(self, frame: np.ndarray) -> np.ndarray:
        """
        Wrapper for backward compatibility - now shows all players

        Returns:
            Frame with health regions highlighted
        """
        return self.visualize_all_players(frame)

    def save_debug_images(self, frame: np.ndarray, output_prefix: str = "health_debug"):
        """Save debug images for health detection"""
        sidebar = self.extract_sidebar_region(frame)

        # Save sidebar
        cv2.imwrite(f"{output_prefix}_sidebar.png", sidebar)

        # Save yellow mask
        hsv = cv2.cvtColor(sidebar, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        cv2.imwrite(f"{output_prefix}_yellow_mask.png", yellow_mask)

        # Try to extract health region
        result = self.extract_health_region(frame)
        if result is not None:
            health_region, info = result
            cv2.imwrite(f"{output_prefix}_health_region.png", health_region)

            # Save processed version
            processed = self.preprocess_for_ocr(health_region)
            cv2.imwrite(f"{output_prefix}_processed.png", processed)

        # Save visualization
        vis_frame = self.visualize_health_detection(frame)
        cv2.imwrite(f"{output_prefix}_visualization.png", vis_frame)


# Example usage and testing
if __name__ == "__main__":
    import os
    import glob

    detector = TFTHealthDetector()

    # Test on screenshots
    test_files = glob.glob("screenshots/*.png")[:5]

    print("Testing Health Detection")
    print("=" * 60)

    for filepath in test_files:
        if os.path.exists(filepath):
            frame = cv2.imread(filepath)
            if frame is not None:
                result = detector.detect_health(frame)

                filename = os.path.basename(filepath)
                print(f"\n{filename}:")
                print(f"  Found: {result['found']}")
                print(f"  Health: {result['health']}")
                print(f"  Confidence: {result['confidence']:.2f}")

                # Save debug images for first file
                if filepath == test_files[0]:
                    detector.save_debug_images(frame, "health_debug")

    print("\nDebug images saved: health_debug_*.png")