import pandas as pd

from src.clustering_solver import ClusteringSolver
from src.k_means_multi_threaded import KMeansMultiThreaded
from src.visualisation import visualisation_3d

PATH = "input"
FILENAME = "Country-data.csv"
K = 3


def get_data(path: str, filename: str) -> pd.DataFrame:
    filepath = path + '/' + filename
    data = pd.read_csv(filepath)
    data_copy = data.copy(True)
    data_copy.drop(columns=["country"], axis=1, inplace=True)

    return data_copy


if __name__ == "__main__":
    data = get_data(PATH, FILENAME)
    solver = ClusteringSolver()
    solver.set_k(K)
    solver.set_algorithm(KMeansMultiThreaded)
    cluster_content = solver.execute(data)

    # for key, value in cluster_content.items():
    #     print(f"{key}: {len(value)}")

    # visualisation_3d(cluster_content)
