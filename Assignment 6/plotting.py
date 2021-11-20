import numpy as np
from matplotlib import pyplot as plt

# Markers to see different clusters
markers = [".", ",", "o", "v", "^", "<", ">", "8", "s", "p", "P", "*", "h", "H", "D", "d", "|", "_", ".", ",", "o", "v",
           "^", "<", ]


# Source: https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
def get_cmap(n, name='hsv'):
    """
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.
    """
    return plt.cm.get_cmap(name, n)


def plot_data_points(data, eps, min_pts):
    fig, ax = plt.subplots()
    ax.title.set_text(f"DBSCAN with MinPts={min_pts} eps={eps}")
    cluster_amount = 1000
    cmap = get_cmap(cluster_amount)
    edge_color = (0, 0, 0, 0.5)
    for p in data:
        cluster = p[2]
        # If noise
        if cluster == -1:
            color = "white"
            marker = "X"
        else:
            color = cmap(cluster*100)
            marker = markers[cluster]
        ax.scatter(p[0], p[1], color=color, marker=marker, edgecolors=edge_color)
    plt.show()


def plot_knn(distances, k):
    # Trick: Plot column 0 of sorted distances because col 0 is the sum of that point's distances
    plt.plot(distances[:, 0])
    plt.title(f"{k}-Nearest Neighbours Plot")
    plt.ylabel("Distance")
    plt.xlabel("Points (sorted by total distance to NN)")
    # Cutoffs for Ks
    #   3 | ~0.055
    #   4 | ~0.075
    #   5 | ~0.08

    # Threshold estimates for elbow method
    if k == 3:
        plt.axhline(0.055, linestyle='--', color='black')
    if k == 4:
        plt.axhline(0.075, linestyle='--', color='black')
    if k == 5:
        plt.axhline(0.08, linestyle='--', color='black')
    plt.show()
