"""Product catalogue (data/products.json)."""
import json
from dataclasses import dataclass
from pathlib import Path

from .render import hex_to_bgr

DEFAULT_CATALOGUE = Path(__file__).resolve().parent.parent / "data" / "products.json"


@dataclass
class Lipstick:
    id: str
    name: str
    hex: str
    coverage: float = 0.6
    finish: str = "satin"   # used from week 8 on

    @property
    def bgr(self):
        return hex_to_bgr(self.hex)


def load_lipsticks(path=DEFAULT_CATALOGUE):
    with open(path, encoding="utf-8") as f:
        return [Lipstick(**p) for p in json.load(f)["lipsticks"]]
