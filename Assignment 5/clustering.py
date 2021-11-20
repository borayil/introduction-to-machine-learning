import numpy as np
from calculation import *
from plotting import *
from matplotlib import pyplot as plt


def plot_dendrograms(link_single, link_average, link_complete, link_ward):
    plot_dendrogram(link_single, title="Dendrogram using Single Linkage Method", cutoffs=[0.15, 0.14, 0.13])
    plot_dendrogram(link_average, title="Dendrogram using Average Linkage Method", cutoffs=[0.4, 0.295, 0.27])
    plot_dendrogram(link_complete, title="Dendrogram using Complete Linkage Method", cutoffs=[0.7, 0.63, 0.6])
    plot_dendrogram(link_ward, title="Dendrogram using Ward Linkage Method", cutoffs=[3.5, 2.2, 1.1])


def plot_clusters(clusters_single, clusters_average, clusters_complete, clusters_ward):
    for i in range(3):
        plot_data_points(data, clusters=clusters_single[i], title=f"K={i + 2} Clusters using the Single Linkage Method")
    for i in range(3):
        plot_data_points(data, clusters=clusters_average[i], title=f"K={i + 2} Clusters using the Average Linkage Method")
    for i in range(3):
        plot_data_points(data, clusters=clusters_complete[i], title=f"K={i + 2} Clusters using the Complete Linkage Method")
    for i in range(3):
        plot_data_points(data, clusters=clusters_ward[i], title=f"K={i + 2} Clusters using the Ward Linkage Method")


def print_silhouette_scores(scores_single, scores_average, scores_complete, scores_ward):
    print("Silhouette Scores Single Linkage:")
    for i in range(3):
        print(f"K={i + 2}:", scores_single[i])
    print("Silhouette Scores Average Linkage:")
    for i in range(3):
        print(f"K={i + 2}:", scores_average[i])
    print("Silhouette Scores Complete Linkage:")
    for i in range(3):
        print(f"K={i + 2}:", scores_complete[i])
    print("Silhouette Scores Ward Linkage:")
    for i in range(3):
        print(f"K={i + 2}:", scores_ward[i])


if __name__ == '__main__':
    # Reading in the Data File
    data = np.loadtxt(open("data/data_clustering.csv", "r+"), delimiter=",")

    # apply different linkage methods
    link_single = single_linkage(data)
    print(link_single)
    link_average = average_linkage(data)
    link_complete = complete_linkage(data)
    link_ward = ward_linkage(data)

    # creating the different clusters for the different methods
    clusters_single = [
        clusters(link_single, 2),
        clusters(link_single, 3),
        clusters(link_single, 4)
    ]
    clusters_average = [
        clusters(link_average, 2),
        clusters(link_average, 3),
        clusters(link_average, 4)
    ]
    clusters_complete = [
        clusters(link_complete, 2),
        clusters(link_complete, 3),
        clusters(link_complete, 4)
    ]
    clusters_ward = [
        clusters(link_ward, 2),
        clusters(link_ward, 3),
        clusters(link_ward, 4)
    ]

    # compute silhouette scores
    scores_single = [
        silhouette_score(data, clusters_single[0]),
        silhouette_score(data, clusters_single[1]),
        silhouette_score(data, clusters_single[2])
    ]
    scores_average = [
        silhouette_score(data, clusters_average[0]),
        silhouette_score(data, clusters_average[1]),
        silhouette_score(data, clusters_average[2])
    ]
    scores_complete = [
        silhouette_score(data, clusters_complete[0]),
        silhouette_score(data, clusters_complete[1]),
        silhouette_score(data, clusters_complete[2])
    ]
    scores_ward = [
        silhouette_score(data, clusters_ward[0]),
        silhouette_score(data, clusters_ward[1]),
        silhouette_score(data, clusters_ward[2])
    ]

    # print report of silhouette scores
    print_silhouette_scores(scores_single, scores_average, scores_complete, scores_ward)
    # plot original data points
    plot_data_points(data, title="Original Data Points")
    # plotting the dendrograms with fitting cut-off thresholds
    plot_dendrograms(link_single, link_average, link_complete, link_ward)
    # plotting different clusters for the different linkage methods
    plot_clusters(clusters_single, clusters_average, clusters_complete, clusters_ward)
    plt.show()
