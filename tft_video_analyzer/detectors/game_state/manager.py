#!/usr/bin/env python3
"""
TFT Game State Manager
Tracks the current state of a TFT game during video processing
"""

from enum import Enum
from typing import Optional, Dict, Tuple

class GamePhase(Enum):
    """Phases of a TFT game"""
    LOADING = "loading"
    PLANNING = "planning"
    COMBAT = "combat"
    CAROUSEL = "carousel"
    AUGMENT = "augment"
    PVE = "pve"
    POSTGAME = "postgame"
    UNKNOWN = "unknown"

class RoundType(Enum):
    """Types of rounds in TFT"""
    UNIVERSE = "universe"  # 1-1
    PVE_MINIONS = "pve_minions"  # 1-2, 1-3, 1-4
    PVE_KRUGS = "pve_krugs"  # 2-7
    PVE_WOLVES = "pve_wolves"  # 3-7
    PVE_RAPTORS = "pve_raptors"  # 4-7
    PVE_ELDER = "pve_elder"  # 5-7
    CAROUSEL = "carousel"  # X-4 rounds
    PVP = "pvp"  # Regular player vs player
    UNKNOWN = "unknown"

class TFTGameState:
    """Manages the state of a TFT game"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the game state to initial values"""
        self.stage = 0
        self.round = 0
        self.phase = GamePhase.UNKNOWN
        self.round_type = RoundType.UNKNOWN
        self.game_started = False
        self.game_ended = False
        self.last_phase = GamePhase.UNKNOWN
        self.phase_start_time = None
        self.round_start_time = None

    def start_game(self):
        """Mark the game as started"""
        self.game_started = True
        self.stage = 1
        self.round = 1
        self.phase = GamePhase.PLANNING
        self.round_type = RoundType.UNIVERSE
        print(f"Game started - Stage {self.stage}-{self.round} ({self.round_type.value})")

    def get_round_type(self, stage: int, round: int) -> RoundType:
        """Determine the type of round based on stage and round number"""

        # Stage 1 special rounds
        if stage == 1:
            if round == 1:
                return RoundType.UNIVERSE
            elif round in [2, 3, 4]:
                return RoundType.PVE_MINIONS
            else:
                return RoundType.UNKNOWN

        # Carousel rounds (X-4)
        if round == 4:
            return RoundType.CAROUSEL

        # PVE rounds (X-7)
        if round == 7:
            if stage == 2:
                return RoundType.PVE_KRUGS
            elif stage == 3:
                return RoundType.PVE_WOLVES
            elif stage == 4:
                return RoundType.PVE_RAPTORS
            elif stage >= 5:
                return RoundType.PVE_ELDER

        # Regular PVP rounds
        return RoundType.PVP

    def update_round(self, stage: int, round: int):
        """Update the current stage and round"""
        if stage != self.stage or round != self.round:
            self.stage = stage
            self.round = round
            self.round_type = self.get_round_type(stage, round)
            self.round_start_time = None  # Reset timer

            # Determine initial phase for the round
            if self.round_type == RoundType.CAROUSEL:
                self.phase = GamePhase.CAROUSEL
            elif self.round_type == RoundType.UNIVERSE:
                # Universe round (1-1) doesn't have planning
                self.phase = GamePhase.PLANNING
            else:
                # All other rounds start with planning
                self.phase = GamePhase.PLANNING

            print(f"Round updated - Stage {self.stage}-{self.round} ({self.round_type.value})")

    def set_phase(self, phase: GamePhase):
        """Update the current phase"""
        if phase != self.phase:
            self.last_phase = self.phase
            self.phase = phase
            self.phase_start_time = None  # Reset timer
            print(f"Phase changed: {self.last_phase.value} -> {self.phase.value}")

    def transition_to_combat(self):
        """Transition from planning to combat phase"""
        if self.phase == GamePhase.PLANNING:
            self.set_phase(GamePhase.COMBAT)

    def transition_to_planning(self):
        """Transition from combat to planning phase"""
        if self.phase == GamePhase.COMBAT:
            self.set_phase(GamePhase.PLANNING)

    def advance_round(self):
        """Advance to the next round"""
        next_stage, next_round = self.get_next_round(self.stage, self.round)

        if next_stage is not None and next_round is not None:
            self.update_round(next_stage, next_round)
        else:
            print("Cannot advance round - game may have ended")

    def get_next_round(self, stage: int, round: int) -> Tuple[Optional[int], Optional[int]]:
        """Calculate the next stage and round"""

        # Special handling for stage 1
        if stage == 1:
            if round < 4:
                return stage, round + 1
            else:
                # Move to stage 2-1
                return 2, 1

        # Standard stages (2+)
        if round < 7:
            return stage, round + 1
        else:
            # Move to next stage
            return stage + 1, 1

    def is_carousel_round(self) -> bool:
        """Check if current round is a carousel round"""
        return self.round == 4 and self.stage >= 2

    def is_pve_round(self) -> bool:
        """Check if current round is a PVE round"""
        return self.round_type in [
            RoundType.PVE_MINIONS,
            RoundType.PVE_KRUGS,
            RoundType.PVE_WOLVES,
            RoundType.PVE_RAPTORS,
            RoundType.PVE_ELDER
        ]

    def is_pvp_round(self) -> bool:
        """Check if current round is a PVP round"""
        return self.round_type == RoundType.PVP

    def get_state_string(self) -> str:
        """Get a formatted string of the current state"""
        if not self.game_started:
            return "Game not started"

        if self.game_ended:
            return "Game ended"

        return f"Stage {self.stage}-{self.round} | {self.phase.value} | {self.round_type.value}"

    def get_state_dict(self) -> Dict:
        """Get the current state as a dictionary"""
        return {
            "stage": self.stage,
            "round": self.round,
            "phase": self.phase.value,
            "round_type": self.round_type.value,
            "game_started": self.game_started,
            "game_ended": self.game_ended,
            "is_carousel": self.is_carousel_round(),
            "is_pve": self.is_pve_round(),
            "is_pvp": self.is_pvp_round(),
            "state_string": self.get_state_string()
        }

    def should_capture_snapshot(self) -> bool:
        """Determine if current state is worth capturing a snapshot"""

        # Capture at the start of each phase
        if self.phase_start_time is None:
            return True

        # Always capture carousel rounds
        if self.phase == GamePhase.CAROUSEL:
            return True

        # Capture universe round (1-1)
        if self.stage == 1 and self.round == 1:
            return True

        # Capture planning and combat phases
        if self.phase in [GamePhase.PLANNING, GamePhase.COMBAT]:
            return True

        return False