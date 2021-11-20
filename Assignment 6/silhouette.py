import math


# Normal Euclidean distance (so we can also compare with assignment 5 silhouette scores)
def distance(_p, p):
    return math.sqrt((_p[0] - p[0]) ** 2 + (_p[1] - p[1]) ** 2)


# Average distance of a point to points within cluster
def calculate_a_i(p, cluster):
    distances = []
    for other_point in cluster:
        if p[0] == other_point[0] and p[1] == other_point[1]: continue
        distances.append(distance(p, other_point))
    return sum(distances) / len(distances)


# Average distance of a point to points within closest cluster
def calculate_b_i(p, closest_cluster):
    distances = []
    for other_point in closest_cluster:
        distances.append(distance(p, other_point))
    return sum(distances) / len(distances)


# Silhouette score 
def calculate_s_i(a, b):
    return (b-a)/max((a,b))


def find_closest_cluster(p, p_cluster, clusters):
    closest_distance = float('inf')
    closest_cluster = None
    clusters.remove(p_cluster)

    for cluster in clusters:
        for other_point in cluster:
            d = distance(p, other_point)
            if d < closest_distance:
                closest_distance = d
                closest_cluster = cluster
    return closest_cluster


def number_is_found(x, arr):
    for val in arr:
        if val == x:
            return True
    return False


def find_cluster_numbers(data):
    cluster_numbers = []
    for p in data:
        if number_is_found(p[2], cluster_numbers):
            continue
        cluster_numbers.append(p[2])
    return cluster_numbers


def calculate_silhouette(data):
    # Silhouette score
    # Create clusters from the data
    clusters = []
    cluster_numbers = find_cluster_numbers(data)
    for cluster_num in cluster_numbers:
        if cluster_num == -1: continue  # skip noise
        c = []
        for p in data:
            if p[2] == cluster_num:
                c.append([p[0], p[1]])
        clusters.append(c)
    # Now, clusters[0..len(clusters] are the clusters.
    # Calculate silhouette score for each data point
    silhouette_scores = []
    for cluster in clusters:
        for p in cluster:
            # Calculate a(i)
            a = calculate_a_i(p, cluster)
            # Calculate b(i)
            closest_cluster = find_closest_cluster(p, cluster, clusters.copy())
            b = calculate_b_i(p, closest_cluster)
            # Calculate s(i)
            s = calculate_s_i(a, b)
            silhouette_scores.append(s)

    # Mean of silhouette scores is the result
    return sum(silhouette_scores) / len(silhouette_scores)
