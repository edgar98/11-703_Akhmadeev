import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_csv(file):
    return pd.read_csv(file)


def draw(dataset):
    fig, ax = plt.subplots()
    bars = ax.bar(dataset['PassengerId'], dataset['Age'])
    for bar, data in zip(bars, dataset['Sex']):
        if data == 'male':
            bar.set_color('b')
        else:
            bar.set_color('r')
    plt.show()


def draw_data_from_csv(file):
    draw(read_csv(file))


if __name__ == '__main__':
    draw_data_from_csv('test.csv')
