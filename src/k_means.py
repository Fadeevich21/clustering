import math
import random
import pandas as pd

from abc import abstractmethod

from src.clustering_algorithm import ClusteringAlgorithm
from src.area import Area
from src.range import Range
from src.dot import Dot


class KMeans(ClusteringAlgorithm):
    _area = Area()
    __dimension = 0

    def __init__(self) -> None:
        super().__init__()
        pass

    def set_dimension(self, dimension: int) -> None:
        self.__dimension = dimension

    def get_dimension(self) -> int:
        return self.__dimension

    def execute(self, data: pd.DataFrame, k: int) -> dict[Dot, list[Dot]]:
        pass

    @abstractmethod
    def _fill_area(self, data: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def _get_cluster_centers(self, number_cluster_centers: int) -> list[Dot]:
        pass

    def _get_cluster_center(self) -> Dot:
        number_ranges = self._get_number_ranges()
        cluster_center = Dot(number_ranges)
        for i in range(number_ranges):
            range_ = self._area.get_range(i)
            value_in_range = self._get_random_value_in_range(range_)
            cluster_center.set_value(i, value_in_range)

        return cluster_center

    def _get_number_ranges(self) -> int:
        return self.__dimension

    @staticmethod
    def _get_random_value_in_range(range_) -> int | float:
        min_ = range_.max_
        max_ = range_.min_
        value_in_range = random.uniform(min_, max_)

        return value_in_range

    @staticmethod
    def _get_range_array(array) -> Range:
        range_ = Range()
        range_.min_ = min(array)
        range_.max_ = max(array)

        return range_

    def _get_dots(self, data: pd.DataFrame) -> list[Dot]:
        dots = []
        number_dots = self._get_number_dots(data)
        for i in range(number_dots):
            dot = self._get_dot(data.iloc[i, :])
            dots.append(dot)

        return dots

    @staticmethod
    def _get_number_dots(data):
        return len(data)

    def _get_dot(self, values) -> Dot:
        dimension = self.get_dimension()
        dot = Dot(dimension)
        for i in range(dimension):
            dot.set_value(i, values[i])

        return dot

    def _dots_distribution(self, dots, cluster_centers) -> dict[Dot, list[Dot]]:
        cluster_content = dict()
        for cluster_center in cluster_centers:
            cluster_content[cluster_center] = []

        for dot in dots:
            cluster_center = self._get_selected_cluster_center(dot, cluster_centers)
            cluster_content[cluster_center].append(dot)

        return cluster_content

    def _get_selected_cluster_center(self, dot, cluster_centers) -> Dot:
        min_distance = float('inf')
        situable_cluster_center = None
        for cluster_center in cluster_centers:
            distance = 0
            dimension = self.get_dimension()
            for i in range(dimension):
                distance += (dot.get_value(i) - cluster_center.get_value(i)) ** 2

            distance = distance ** (1 / 2)
            if distance < min_distance:
                min_distance = distance
                situable_cluster_center = cluster_center

        return situable_cluster_center

    def _prepare_cluster_centers(self, cluster_content) -> list[Dot]:
        cluster_centers_prepared = []
        dimension = self.get_dimension()
        for _, dots in cluster_content.items():
            cluster_center_prepared = Dot(dimension)
            for i in range(dimension):
                if len(dots) == 0:
                    continue

                num = sum([dot.get_value(i) for dot in dots]) / len(dots)
                cluster_center_prepared.set_value(i, num)
            cluster_centers_prepared.append(cluster_center_prepared)

        return cluster_centers_prepared

    def _centers_is_equals(self, cluster_centers_prepared, cluster_centers):
        dimension = self.get_dimension()
        for i in range(len(cluster_centers)):
            for j in range(dimension):
                if not math.isclose(cluster_centers_prepared[i].get_value(j), cluster_centers[i].get_value(j)):
                    return False

        return True
