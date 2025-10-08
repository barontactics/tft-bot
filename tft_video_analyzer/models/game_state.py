"""
TFT Game State Model

Tracks the current state of a TFT game including gold, health, streak, stage, round,
shop, augments, and units.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class GameState:
    """Represents the current state of a TFT game"""

    # Core game metrics
    gold: Optional[int] = None
    health: Optional[int] = None
    streak: Optional[int] = None  # Positive for win streak, negative for loss streak
    stage: Optional[int] = None
    round: Optional[int] = None

    # Game elements
    shop: List[str] = field(default_factory=list)  # List of champion names in shop
    augments: List[str] = field(default_factory=list)  # List of augment names
    units: List[Dict[str, Any]] = field(default_factory=list)  # List of units on board/bench

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    frame_number: Optional[int] = None

    def __str__(self) -> str:
        """String representation of game state"""
        parts = []
        if self.stage is not None and self.round is not None:
            parts.append(f"Stage {self.stage}-{self.round}")
        if self.health is not None:
            parts.append(f"HP: {self.health}")
        if self.gold is not None:
            parts.append(f"Gold: {self.gold}")
        if self.streak is not None:
            streak_type = "W" if self.streak > 0 else "L"
            parts.append(f"Streak: {abs(self.streak)}{streak_type}")
        if self.shop:
            parts.append(f"Shop: {len(self.shop)} units")
        if self.augments:
            parts.append(f"Augments: {len(self.augments)}")
        if self.units:
            parts.append(f"Units: {len(self.units)}")

        return " | ".join(parts) if parts else "Empty GameState"

    def to_dict(self) -> dict:
        """Convert game state to dictionary"""
        return {
            "gold": self.gold,
            "health": self.health,
            "streak": self.streak,
            "stage": self.stage,
            "round": self.round,
            "shop": self.shop,
            "augments": self.augments,
            "units": self.units,
            "timestamp": self.timestamp.isoformat(),
            "frame_number": self.frame_number
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'GameState':
        """Create game state from dictionary"""
        state = cls(
            gold=data.get("gold"),
            health=data.get("health"),
            streak=data.get("streak"),
            stage=data.get("stage"),
            round=data.get("round"),
            shop=data.get("shop", []),
            augments=data.get("augments", []),
            units=data.get("units", []),
            frame_number=data.get("frame_number")
        )

        if "timestamp" in data:
            state.timestamp = datetime.fromisoformat(data["timestamp"])

        return state

    def is_complete(self) -> bool:
        """Check if all core fields are populated"""
        return all([
            self.gold is not None,
            self.health is not None,
            self.streak is not None,
            self.stage is not None,
            self.round is not None
        ])

    def update(self, **kwargs) -> 'GameState':
        """Update game state fields and return self for chaining"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def add_shop_unit(self, unit_name: str) -> 'GameState':
        """Add a unit to the shop"""
        self.shop.append(unit_name)
        return self

    def add_augment(self, augment_name: str) -> 'GameState':
        """Add an augment"""
        if augment_name not in self.augments:
            self.augments.append(augment_name)
        return self

    def add_unit(self, unit: Dict[str, Any]) -> 'GameState':
        """Add a unit to the board/bench"""
        self.units.append(unit)
        return self

    def clear_shop(self) -> 'GameState':
        """Clear the shop (e.g., after reroll)"""
        self.shop = []
        return self
