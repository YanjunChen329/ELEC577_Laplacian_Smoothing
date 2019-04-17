import numpy as np
import matplotlib.pyplot as plt


def plot_batchsize():
    ls_sgd = np.array([96.21, 98.45, 98.72, 99.01, 99.42, 99.38]) / 100.
    sgd = np.array([11.35, 92.74, 94.89, 97.05, 98.56, 98.92]) / 100.

    index = np.arange(ls_sgd.shape[0])
    bar_width = 0.3
    plt.bar(index, sgd, bar_width, color='navy', label="SGD", alpha=0.8, linewidth=0.6, edgecolor='black')
    plt.bar(index + bar_width, ls_sgd, bar_width, color='yellow', label="LS-SGD", alpha=0.8, linewidth=0.6, edgecolor='black')

    plt.xlabel("Batch Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.ylim(top=1)
    plt.xticks(index + bar_width/2., ("2", "4", "8", "16", "32", "64"))
    plt.show()

if __name__ == '__main__':
    plot_batchsize()