from dataclasses import dataclass


@dataclass(slots=True)
class RankedItem:
    passage_id: str
    rank: int
    score: float
