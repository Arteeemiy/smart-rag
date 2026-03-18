from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievedChunk:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    distance: float = 0.0
