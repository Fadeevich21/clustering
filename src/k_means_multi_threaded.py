import pandas as pd
import threading
from threading import Thread
import copy
import csv

from abc import ABC

from src.k_means import KMeans
from src.range import Range
from src.dot import Dot


class KMeansMultiThreaded(KMeans, ABC):
    __number_threads = 2
    __number_using_thread = 0
    __csv_writer_lock = threading.Lock()

    def __init__(self) -> None:
        super().__init__()

    def execute(self, data: pd.DataFrame, k: int):
        self.set_dimension(len(data.iloc[0, :]))
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

    def __fill_area_multi_thread(self, data, number_thread) -> None:
        self.__number_using_thread += 1
        with open('tmp/area.csv', 'a', newline='') as area_file:
            area_writer = csv.writer(area_file)
            arr = [number_thread]
            for i in range(self._get_number_dots(data.columns)):
                range_ = self._get_range_array(data.iloc[:, i])
                arr.append(range_.min_)
                arr.append(range_.max_)

            with self.__csv_writer_lock:
                area_writer.writerow(arr)

        self.__number_using_thread -= 1

    def _fill_area(self, data: pd.DataFrame) -> None:
        self.__clear_area_file()
        step = len(data) // self.__number_threads
        for i in range(self.__number_threads):
            if i + 1 == self.__number_threads:
                data_thread = data.iloc[i*step:, :]
            else:
                data_thread = data.iloc[i*step:(i + 1)*step:, :]
            th = Thread(target=self.__fill_area_multi_thread, args=(data_thread, i+1))
            th.start()

        self.__wait_end_work_threads()
        self.__prepare_area_file()

    @staticmethod
    def __clear_area_file():
        file = open("tmp/area.csv", 'w')
        file.close()

    def __wait_end_work_threads(self):
        while self.__number_using_thread != 0:
            pass

    def __prepare_area_file(self):
        area_data = pd.read_csv("tmp/area.csv")
        dimensions = self.get_dimension()
        with open('tmp/area.csv', 'w', newline='') as area_file:
            area_writer = csv.writer(area_file)
            arr = [0]
            for i in range(dimensions):
                range_ = Range()
                range_.min_ = min(area_data.iloc[:, i*2 + 1])
                range_.max_ = max(area_data.iloc[:, i*2 + 2])
                arr.append(range_.min_)
                arr.append(range_.max_)

            area_writer.writerow(arr)

    def _get_cluster_centers(self, number_cluster_centers: int) -> list[Dot]:
        with open("tmp/centers.csv", 'w') as centers_file:
            cluster_centers = []
            centers_writer = csv.writer(centers_file)
            for i in range(number_cluster_centers):
                arr = [i+1]
                cluster_center = self._get_cluster_center()
                cluster_centers.append(cluster_center)
                for j in range(len(cluster_center)):
                    arr.append(cluster_center.get_value(j))
                centers_writer.writerow(arr)

        return cluster_centers
