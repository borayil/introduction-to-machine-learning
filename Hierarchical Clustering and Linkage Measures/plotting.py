from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt


def plot_dendrogram(data, x_lab="Data Points", y_lab="Proximity Measure", title="", cutoffs=[]):
    plt.figure()
    plt.title(title)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    for val in cutoffs:
        plt.axhline(val, linestyle='--', color='black')
    dendrogram(data)


def plot_data_points(data, x_lab="Feature Space of Feature 1", y_lab="Feature Space of Feature 2", title="", clusters=None):
    plt.figure()
    plt.title(title)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    edge_color = (0, 0, 0, 0.5)
    if clusters is None:
        plt.scatter(data[:, 0], data[:, 1], color='gray', edgecolors=edge_color)
    else:
        plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='hsv', edgecolors=edge_color)
