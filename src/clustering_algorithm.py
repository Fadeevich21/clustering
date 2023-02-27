import pandas as pd

from abc import ABC, abstractmethod


class ClusteringAlgorithm(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def execute(self, data: pd.DataFrame, k: int):
        pass
