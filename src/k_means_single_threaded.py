import pandas as pd
import copy

from abc import ABC

from src.k_means import KMeans
from src.dot import Dot


class KMeansSingleThreaded(KMeans, ABC):

    def __init__(self) -> None:
        super().__init__()

    def execute(self, data: pd.DataFrame, k: int):
        self.set_dimension(len(data.iloc[0, :]))
        self._area.clear()
        self._fill_area(data)
        dots = self._get_dots(data)
        cluster_centers = self._get_cluster_centers(k)
        while 1:
            cluster_content = self._dots_distribution(dots, cluster_centers)
            cluster_centers_prepared = self._prepare_cluster_centers(cluster_content)
            if self._centers_is_equals(cluster_centers_prepared, cluster_centers):
                break

            cluster_centers = copy.deepcopy(cluster_centers_prepared)

        return cluster_content

    def _fill_area(self, data: pd.DataFrame) -> None:
        for i in range(self._get_number_dots(data.columns)):
            range_ = self._get_range_array(data.iloc[:, i])
            self._area.add_range(range_)

    def _get_cluster_centers(self, number_cluster_centers: int) -> list[Dot]:
        cluster_centers = []
        for i in range(number_cluster_centers):
            cluster_center = self._get_cluster_center()
            cluster_centers.append(cluster_center)

        return cluster_centers
