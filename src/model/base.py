from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict
from PIL import Image

@dataclass
class ItemData:
    name: str
    count: int
    total_price: float
    id: int = field(init=False) 

    def __post_init__(self):
        self.id = abs(hash(self.name + str(self.total_price)))

@dataclass
class ReceiptData:
    items: Dict[int, ItemData]
    total: float

class AIModel(ABC):
    @abstractmethod
    def run(self, image: Image.Image) -> ReceiptData:
        pass