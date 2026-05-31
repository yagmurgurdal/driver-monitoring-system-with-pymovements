from dataclasses import dataclass
from typing import Callable

from sklearn.base import ClassifierMixin


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    builder: Callable[[int, int], ClassifierMixin]
    use_balanced_sample_weight: bool = False
