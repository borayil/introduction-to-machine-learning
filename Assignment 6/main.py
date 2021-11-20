from dbscan import *
from knn import *
from plotting import *
from silhouette import *
import copy


def prepare_data(data):
    for p in data:
        p.append(None)
        p.append(False)


if __name__ == "__main__":
    # Read data
    path = "data/data_clustering.csv"
    read_data = np.loadtxt(open(path, "r+"), delimiter=",")
    read_data = read_data.tolist()
    data = copy.deepcopy(read_data)

    # Prepare data format
    prepare_data(data)

    # K-nearest neighbors search
    k_nearest_neighbours(read_data)

    # Parameters
    # (Min pts) | (eps estimate / guess from K-NN search)
    #   3 | ~0.055
    #   4 | ~0.075
    #   5 | ~0.08
    all_min_pts = [3,4,5]
    epsilons = [0.055, 0.075, 0.08]
    
    # For each min_pts and eps combination, run DBSCAN and display results.
    for i in range(len(all_min_pts)):
        # Initialize data and parameters
        dbscan_data = copy.deepcopy(data)
        min_pts = all_min_pts[i]
        eps = epsilons[i]

        # Call dbscan
        dbscan(dbscan_data, eps, min_pts)
    
        # Results and Plots
        plot_data_points(dbscan_data, eps, min_pts)
        
        # Silhouette Score
        s = calculate_silhouette(dbscan_data)
        print(f"Silhouette score with min_pts = {min_pts} and eps = {eps} -> {s}")
        
    
    