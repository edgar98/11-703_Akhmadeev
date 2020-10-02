import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_csv(file):
    return pd.read_csv(file)


def draw_data_from_csv(file):
    draw(read_csv(file))


def exp_smoothing(input_data, alpha=0.):
    output = [input_data[0]]
    for i in range(1, len(input_data)):
        output.append(alpha * input_data[i] + (1 - alpha) * output[i - 1])
    return output


def draw(dataset):
    date = list(dataset.Date)
    sunspot = list(dataset['Monthly Mean Total Sunspot Number'])
    alpha = 0.1
    sunspot_new = exp_smoothing(sunspot, alpha=alpha)
    plt.figure(figsize=(30, 10))
    plt.plot(date, sunspot_new, color='r')
    plt.scatter(date, sunspot)
    plt.title('alpha = ' + str(alpha))
    plt.show()


if __name__ == '__main__':
    draw_data_from_csv('Sunspots.csv')
