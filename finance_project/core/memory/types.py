# app/core/memory/types.py

from typing import Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class Memory:
    id: str
    type: str                 # "preference", "emotional", "behavioral", etc.
    content: str
    confidence: float         # 0.0 – 1.0
    importance: float         # 0.0 – 1.0
    exposure_count: int = 0
    last_used_at: Optional[datetime] = None

    def decay(self):
        """
        Reduces influence after repeated exposure.
        """
        self.confidence = max(0.2, self.confidence - 0.1)
        self.importance = max(0.2, self.importance - 0.1)