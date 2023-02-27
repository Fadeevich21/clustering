import pandas as pd

from src.clustering_algorithm import ClusteringAlgorithm


class ClusteringSolver:
    __k = 0
    __algorithm: ClusteringAlgorithm = None

    def __init__(self):
        pass

    def set_k(self, k: int) -> None:
        self.__k = k

    def set_algorithm(self, algorithm: ClusteringAlgorithm) -> None:
        self.__algorithm = algorithm()

    def execute(self, data: pd.DataFrame):
        return self.__algorithm.execute(data, self.__k)
