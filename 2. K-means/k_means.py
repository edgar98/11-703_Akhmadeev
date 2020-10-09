from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def generate_random_data():
    # Set three centers, the model should predict similar results
    center_1 = np.array([1, 1])
    center_2 = np.array([5, 5])
    center_3 = np.array([8, 1])

    # Generate random data and center it to the three centers
    data_1 = np.random.randn(200, 2) + center_1
    data_2 = np.random.randn(200, 2) + center_2
    data_3 = np.random.randn(200, 2) + center_3

    data = np.concatenate((data_1, data_2, data_3), axis=0)

    plt.scatter(data[:, 0], data[:, 1], s=7)
    plt.show()
    return data


def find_k(data):
    pass


def generate_random_centers(k, c):
    # Generate random centers, here we use sigma and mean to ensure it represent the whole data
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    centers = np.random.randn(k, c) * std + mean
    return centers


def k_means_algorithm(data):
    # Number of clusters
    k = 3
    # k = find_k(data)
    # Number of training data
    n = data.shape[0]
    # Number of features in the data
    c = data.shape[1]

    centers = generate_random_centers(k, c)

    # Plot the data and the centers generated as random
    plt.scatter(data[:, 0], data[:, 1], s=7)
    plt.scatter(centers[:, 0], centers[:, 1], marker='*', c='g', s=150)
    plt.show()
    centers_old = np.zeros(centers.shape)  # to store old centers
    centers_new = deepcopy(centers)  # Store new centers

    clusters = np.zeros(n)
    distances = np.zeros((n, k))

    error = np.linalg.norm(centers_new - centers_old)

    # When, after an update, the estimate of that center stays the same, exit loop
    while error != 0:
        # Measure the distance to every center
        for i in range(k):
            distances[:, i] = np.linalg.norm(data - centers[i], axis=1)
        # Assign all training data to closest center
        clusters = np.argmin(distances, axis=1)

        centers_old = deepcopy(centers_new)
        # Calculate mean for every cluster and update the center
        for i in range(k):
            centers_new[i] = np.mean(data[clusters == i], axis=0)
        error = np.linalg.norm(centers_new - centers_old)

    # Plot the data and the centers generated as random
    plt.scatter(data[:, 0], data[:, 1], s=7)
    plt.scatter(centers_new[:, 0], centers_new[:, 1], marker='*', c='g', s=150)
    plt.show()
    return centers


if __name__ == '__main__':
    data = generate_random_data()
    k_means_algorithm(data)
