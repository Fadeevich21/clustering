import matplotlib.pyplot as plt

from src.dot import Dot


def visualisation_3d(cluster_content: dict[Dot, list[Dot]]):
    ax = plt.axes(projection="3d")
    plt.xlabel("x")
    plt.ylabel("y")

    for cluster_center, dots in cluster_content.items():
        x_coordinates = []
        y_coordinates = []
        z_coordinates = []
        for dot in dots:
            x_coordinates.append(dot.get_value(0))
            y_coordinates.append(dot.get_value(1))
            z_coordinates.append(dot.get_value(2))
        ax.scatter(x_coordinates, y_coordinates, z_coordinates)

        ax.scatter(cluster_center.get_value(0), cluster_center.get_value(1), cluster_center.get_value(2), color='red')

    plt.show()
