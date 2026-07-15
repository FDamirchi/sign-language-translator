from dataclasses import dataclass

@dataclass(frozen=True, slots=True) 
class Prediction:
    label: str
    confidence: float

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Prediction label cannot be empty!")
        
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Prediction confidence must be between 0.0 and 1.0!")