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
    __area_filename = "tmp/area.csv"
    __centers_filename = "tmp/centers.csv"
    __nclusters_filename = "tmp/nclusters.csv"
    __nmeans_filename = "tmp/nmeans.csv"

    __number_threads = 2
    __number_using_thread = 0
    __csv_writer_lock = threading.Lock()

    def __init__(self) -> None:
        super().__init__()

    def execute(self, data: pd.DataFrame, k: int):
        self.__clear_area_file()
        self.set_dimension(len(data.iloc[0, :]))
        self._fill_area(data)
        dots = self._get_dots(data)
        self._fill_cluster_centers(k)
        while 1:
            self.__clear_nclusters_file()
            self.__clear_nmeans_file()
            self._dots_distribution_multi_thread(dots)
            # if self._centers_is_equals(cluster_centers_prepared):
            break

    def __fill_area_multi_thread(self, data, number_thread) -> None:
        self.__number_using_thread += 1
        with open(self.__area_filename, 'a', newline='') as area_file:
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
        step = len(data) // self.__number_threads
        for i in range(self.__number_threads):
            if i + 1 == self.__number_threads:
                data_thread = data.iloc[i * step:, :]
            else:
                data_thread = data.iloc[i * step:(i + 1) * step, :]
            th = Thread(target=self.__fill_area_multi_thread, args=(data_thread, i + 1))
            th.start()

        self.__wait_end_work_threads()
        self.__prepare_area_file()

    def __clear_area_file(self):
        self.__clear_file(self.__area_filename)

    def __clear_file(self, filename):
        file = open(filename, 'w')
        file.close()

    def __wait_end_work_threads(self):
        while self.__number_using_thread != 0:
            pass

    def __prepare_area_file(self):
        dimensions = self.get_dimension()
        arr = [0]
        for i in range(dimensions):
            range_ = self.__get_range(i)
            arr.append(range_.min_)
            arr.append(range_.max_)

        with open(self.__area_filename, 'w', newline='') as area_file:
            area_writer = csv.writer(area_file)
            area_writer.writerow(arr)

    def _fill_cluster_centers(self, number_cluster_centers: int) -> None:
        with open(self.__centers_filename, 'w') as centers_file:
            centers_writer = csv.writer(centers_file)
            for i in range(number_cluster_centers):
                arr = [i + 1]
                cluster_center = self._get_cluster_center_multi_thread()
                for j in range(len(cluster_center)):
                    arr.append(cluster_center.get_value(j))
                centers_writer.writerow(arr)

    @staticmethod
    def __get_key(d, value):
        for k, v in d.items():
            if v == value:
                return k

    def __dots_distribution_thread(self, dots, number_thread) -> None:
        self.__number_using_thread += 1

        cluster_centers_map = self._get_cluster_centers_from_file()
        cluster_content: dict[Dot, list[Dot]] = dict()
        for _, cluster_center in cluster_centers_map.items():
            cluster_content[cluster_center] = []

        for dot in dots:
            cluster_centers = cluster_centers_map.values()
            cluster_center = self._get_selected_cluster_center(dot, cluster_centers)
            cluster_content[cluster_center].append(dot)

        with open(self.__nclusters_filename, 'a') as nclusters_file:
            nclusters_writer = csv.writer(nclusters_file)
            dimension = self.get_dimension()
            for cluster_center, dots_cluster in cluster_content.items():
                number_cluster = self.__get_key(cluster_centers_map, cluster_center)
                arr = [number_thread, number_cluster]
                for dot_cluster in dots_cluster:
                    for i in range(dimension):
                        arr.append(dot_cluster.get_value(i))

                with self.__csv_writer_lock:
                    nclusters_writer.writerow(arr)

        with open(self.__nmeans_filename, 'a') as nmeans_file:
            nmeans_writer = csv.writer(nmeans_file)
            for cluster_center, dots_cluster in cluster_content.items():
                number_cluster = self.__get_key(cluster_centers_map, cluster_center)
                arr = [number_thread, number_cluster, len(dots_cluster)]
                for i in range(dimension):
                    if len(dots_cluster) == 0:
                        continue

                    num = sum([dot_cluster.get_value(i) for dot_cluster in dots_cluster]) / len(dots_cluster)
                    arr.append(num)
                with self.__csv_writer_lock:
                    nmeans_writer.writerow(arr)

        self.__number_using_thread -= 1

    def _dots_distribution_multi_thread(self, dots) -> None:
        step = len(dots) // self.__number_threads
        for i in range(self.__number_threads):
            if i + 1 == self.__number_threads:
                data_thread = dots[i * step:]
            else:
                data_thread = dots[i * step:(i + 1) * step]
            th = Thread(target=self.__dots_distribution_thread, args=(data_thread, i + 1))
            th.start()

        self.__wait_end_work_threads()
        self.__prepare_area_file()



    def _get_cluster_centers(self):
        pass

    @staticmethod
    def _get_cluster_centers_from_file():
        cluster_centers = {}
        with open('tmp/centers.csv', 'r') as centers_file:
            for line in centers_file:
                arr = line.split(',')
                number_cluster = int(arr[0])
                arr = list(map(float, arr[1:]))
                cluster_center = Dot(len(arr))
                for i in range(len(arr)):
                    cluster_center.set_value(i, arr[i])
                cluster_centers[number_cluster] = cluster_center

        return cluster_centers

    def _get_cluster_center_multi_thread(self):
        number_ranges = self._get_number_ranges()
        cluster_center = Dot(number_ranges)
        for i in range(number_ranges):
            range_ = self.__get_range(i)
            value_in_range = self._get_random_value_in_range(range_)
            cluster_center.set_value(i, value_in_range)

        return cluster_center

    def __get_range(self, i):
        area_file = open(self.__area_filename, 'r')
        range_ = Range()
        area_data = list(map(float, area_file.readline().split(',')))
        range_.min_ = area_data[i * 2 + 1]
        range_.max_ = area_data[i * 2 + 2]

        return range_

    def __clear_nclusters_file(self):
        self.__clear_file(self.__nclusters_filename)

    def __clear_nmeans_file(self):
        self.__clear_file(self.__nmeans_filename)
