from abc import ABC, classmethod

from PIL import Image

class AIModel(ABC):
    @classmethod
    def run(self, image : Image.Image):
        pass