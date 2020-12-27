import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans


def predict(features, weights):
    zz = np.dot(features, weights)
    return sigmoid(zz)


def sigmoid(zz):
    num = 1 / (1 + np.exp(-zz))
    return num


def update_weights(features, labels, weights, lr):
    N = len(features)
    predictions = predict(features, weights)

    gradient = np.dot(np.transpose(features), predictions - labels)
    gradient /= N
    gradient *= lr

    weights -= gradient
    return weights


def train(features, labels, weights, lr, iters):
    for i in range(iters):
        weights = update_weights(features, labels, weights, lr)

    return weights


def decision(procent):
    return 'blue' if procent >= 0.5 else 'red'


def log(weighss, points):
    print("A: ", weighss[0], "  B: ", weighss[1], "  C: ", weighss[2], )
    print("Новая точка:", points[(len(points) - 1)])
    print("Предсказание новой точки:", decision(predict(points, weighss)[(len(points) - 1)]),
          "% =: ", '{:0.9f}'.format(predict(points, weighss)[(len(points) - 1)] * 100))


def show_figure(x, y, z, xx, yx, zz, colors):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color=colors)))
    fig.add_trace(go.Surface(x=xx, y=yx, z=zz(xx, yx), opacity=0.9, colorscale='Viridis'))
    fig.show()


if __name__ == '__main__':
    # Рандомно задаем точки
    n = 50

    # coordinates (3d)
    x = np.random.randint(0, 100, n + 1)
    y = np.random.randint(0, 100, n + 1)
    z = np.random.randint(0, 100, n + 1)

    # Должны разделить точки на классы, применим для этого k-means
    points = []

    for i in range(n):
        points.append([x[i], y[i], z[i]])

    kmeans = KMeans(n_clusters=2, random_state=0).fit(points)
    clusters = kmeans.labels_

    colors = ['red'] * n

    for i in range(n):
        if clusters[i] == 1:
            colors[i] = 'blue'

    # Добавление новой точки которую нужно будет предсказать
    x_new = np.random.randint(0, 100)
    y_new = np.random.randint(0, 100)
    z_new = np.random.randint(0, 100)
    points.append([x_new, y_new, z_new])
    x[len(x) - 1] = x_new
    y[len(y) - 1] = y_new
    z[len(z) - 1] = z_new
    colors.append('green')

    weighss = train(points[:(len(points) - 1)], clusters, [1, 1, 1], 0.001, 10000)
    log(weighss, points)

    zz = lambda x, y: ((weighss[0] * x - weighss[1] * y) / weighss[2])

    temp = np.linspace(0, 100, 50)
    xx, yx = np.meshgrid(temp, temp)

    show_figure(x, y, z, xx, yx, zz, colors)
