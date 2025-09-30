#!/usr/bin/env python3
"""Example usage of TFT detectors"""

import cv2
from detectors import (
    TFTAugmentDetector,
    TFTLoadingScreenDetector,
    TFTPlanningDetector,
    TFTGameState
)

def main():
    # Initialize detectors
    augment_detector = TFTAugmentDetector()
    loading_detector = TFTLoadingScreenDetector()
    planning_detector = TFTPlanningDetector()
    game_state = TFTGameState()

    # Example: Process a single frame
    frame = cv2.imread("screenshots/augment-selection.png")

    if frame is not None:
        # Check what type of screen it is
        if augment_detector.detect_augment(frame):
            print("Augment selection screen detected!")
            cards = augment_detector.get_augment_cards(frame)
            if cards:
                print(f"Found {len(cards)} augment cards")

        elif loading_detector.detect_loading_screen(frame):
            print("Loading screen detected!")

        elif planning_detector.detect_planning_phase(frame):
            print("Planning phase detected!")

        else:
            print("Unknown game state")

    # Example: Track game state
    print(f"\nCurrent game state: {game_state.get_state()}")

    # Update game state based on detection
    game_state.update_state(frame)
    print(f"Updated game state: {game_state.get_state()}")

if __name__ == "__main__":
    main()