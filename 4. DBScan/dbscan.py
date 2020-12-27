from sklearn.datasets.samples_generator import make_blobs, make_moons
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
import math


def euclidean_distance(point1, point2):
    """
     Возвращает евклидово расстояние между двумя точками.

     Предполагает, что точки представляют собой список значений координат, работает с любым количеством измерений.
    """
    sum = 0
    for i in range(len(point1)):
        sum += (point2[i] - point1[i]) ** 2
    return math.sqrt(sum)


# DBSCAN использует вспомогательную функцию, которая возвращает все точки на определенном расстоянии от данной точки.
def range_query(collection, point, dist_func, eps):
    """
     Находит все точки в коллекции на расстоянии eps от точки.

     Возвращает список значений индекса, указывающих эти точки в коллекции.

     Аргументы:
       collection (iterable): список точек
       point (iterable): точка, представленная в виде списка координат
       distFunc (callable): функция расстояния для использования
       eps (float): epsilon, максимальное расстояние для включения точек
    """
    neighbors = []
    for i in range(len(collection)):
        if dist_func(collection[i], point) < eps:
            neighbors.append(i)

    return neighbors


# Основная функция DBSCAN
def dbscan_algorithm(X, eps, min_pts, dist_func="euclidean"):
    """
    Кластеризация на основе плотности возвращает список меток, назначающих каждую точку кластеру.
    Кластеры будут помечены как 0 и выше, а точки выбросов будут помечены как -1.

    Аргументы:
      X (iterable): набор точек

      eps (float): epsilon, расстояние для точек, которые считаются близкими друг к другу

      minPts (int): количество точек, которые должны быть в пределах эпсилон-расстояния, чтобы точка считалась основной точкой

      distFunc (callable): используемая функция расстояния, по умолчанию используется евклидово расстояние
    """
    if dist_func == "euclidean":
        dist_func = euclidean_distance

    labels = [None] * len(X)  # начать со всех меток undefined
    cluster = 0  # cluster counter

    for i in range(len(X)):
        if labels[i] is not None:
            # Эта точка уже найдена как сосед, пропустите ее
            continue

        # Найти все точки на расстоянии eps до этой точки
        neighbors = range_query(X, X[i], dist_func, eps)

        if len(neighbors) < min_pts:
            # недостаточно connected points, i не является основной точкой
            labels[i] = -1
            continue

        # эта точка находится в кластере
        labels[i] = cluster

        # так как i центральная точка, все соседи находятся в этом кластере
        # найти этих соседей, и, если они также являются ключевыми точками, добавить их соседей
        j = 0
        while j < len(neighbors):
            p = neighbors[j]  # получить индекс этой точки
            if labels[p] == -1:
                # назначить выброс этому кластеру (граничная точка)
                labels[p] = cluster

            if labels[p] is not None:
                # точка уже найдена, пропустите ее
                j += 1
                continue

            # назначить эту точку кластеру и найти его соседей
            labels[p] = cluster
            new_neighbors = range_query(X, X[p], dist_func, eps)
            if len(new_neighbors) >= min_pts:
                # j также является центральной точкой, добавьте его соседей в список для рассмотрения
                # добавляем каждого соседа, которого еще нет в списке, сохраняем порядок в списке
                for n in new_neighbors:
                    if n not in neighbors:
                        neighbors.append(n)

            j += 1  # перейти к следующей точке

        cluster += 1  # увеличить счетчик кластера

    return labels


if __name__ == '__main__':
    blobs_X, blobs_y = make_blobs(n_samples=100, centers=3, n_features=2, cluster_std=2, random_state=42)

    sns.scatterplot(blobs_X[:, 0], blobs_X[:, 1], hue=blobs_y)
    plt.title("Input data")
    plt.show()

    clusters = dbscan_algorithm(blobs_X, eps=2, min_pts=5)
    print(clusters)

    sns.scatterplot(blobs_X[:, 0], blobs_X[:, 1], hue=clusters)
    plt.title("DBSCAN realisation result")
    plt.show()

    # SKlearn DBSCAN
    sklearn_clusters = DBSCAN(eps=2, min_samples=5).fit(blobs_X)
    sns.scatterplot(blobs_X[:, 0], blobs_X[:, 1], hue=sklearn_clusters.labels_)
    plt.title("SKlearn DBSCAN result")
    plt.show()
