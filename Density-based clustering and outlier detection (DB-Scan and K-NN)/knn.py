from sklearn.neighbors import NearestNeighbors
import numpy as np
from plotting import plot_knn


def sort_distances(distances):
    # Use column 0 as sum of columns
    # Because column 0 is always 0 for all (nearest neighbour is itself)
    for distance in distances:
        distance[0] = sum(distance[1:len(distance)])
    return distances[np.argsort(distances[:, 0])]


# 3 plots of knn search for k = 3,4,5 with marked thresholds at elbow points
def k_nearest_neighbours(data):
    # Obtain distance matrix for k
    # Use k = 4,5,6 because itself is counted as a closest neighbour when using sci-learn 
    # Should we also?
    for k in range(3, 6):
        neighbours = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data)
        distances, indices = neighbours.kneighbors(data)
        # Sort in ascending order
        sorted_distances = sort_distances(distances)
        # sorted_distances = sorted_distances[:,1:] removes 0th column
        # Plot
        plot_knn(sorted_distances, k)
