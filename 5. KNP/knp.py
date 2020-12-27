# Алгоритм кратчайшего незамкнутого пути (КНП)
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def first_connection(connection):
    minim = matrix[0][1]
    i_min, j_min = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            if minim > matrix[i][j]:
                minim = matrix[i][j]
                i_min, j_min = i, j
    connection[i_min][j_min] = connection[j_min][i_min] = 1
    connection[i_min][i_min] = connection[j_min][j_min] = -1
    return connection


def connect_all(connection):
    minim = None
    i_min, j_min = 0, 1
    for i in range(n):
        if connection[i][i] == -1:
            for j in range(n):
                if connection[j][j] == 0:
                    if minim is None or minim > connection[i][j]:
                        minim = connection[i][j]
                        i_min, j_min = i, j
    connection[i_min][j_min] = connection[j_min][i_min] = 1
    connection[i_min][i_min] = connection[j_min][j_min] = -1
    return connection


# Обнуляет половину связей
def delete_connection(connection):
    maxim = 0
    i_max, j_max = -1, -1
    for i in range(n):
        for j in range(i + 1, n):
            if connection[i][j] == 1:
                if matrix[i][j] > maxim:
                    maxim = matrix[i][j]
                    i_max, j_max = i, j
    connection[i_max][j_max] = 0
    connection[j_max][i_max] = 0
    return connection


def draw_graph(connection, matrix, n):
    G = nx.from_numpy_matrix(matrix, create_using=nx.DiGraph)
    layout = nx.spring_layout(G)
    nx.draw(G, layout, node_color='lightgreen', edge_color='b')
    nx.draw_networkx_edge_labels(G, pos=layout)
    plt.show()

    weighted_graph = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if connection[i][j] != -1:
                weighted_graph[i][j] = connection[i][j] * matrix[i][j]
            else:
                weighted_graph[i][j] = -1

    G = nx.from_numpy_matrix(weighted_graph, create_using=nx.DiGraph)
    layout = nx.spring_layout(G)
    nx.draw(G, layout, node_color='lightgreen', edge_color='b')
    nx.draw_networkx_edge_labels(G, pos=layout)
    plt.show()


def KNP_alg(matrix, n, k):
    connection = np.zeros((n, n))
    connection = first_connection(connection)
    for i in range(n - 2):
        connection = connect_all(connection)

    for i in range(k - 1):
        connection = delete_connection(connection)

    print("Вывод кратчайшего пути в графе")
    print(connection)

    draw_graph(connection, matrix, n)


if __name__ == '__main__':
    # n - кол-во точек, k - класт.
    n, k = 5, 2
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j] = matrix[j][i] = np.random.randint(1, 100)
    print(matrix)

    KNP_alg(matrix, n, k)
