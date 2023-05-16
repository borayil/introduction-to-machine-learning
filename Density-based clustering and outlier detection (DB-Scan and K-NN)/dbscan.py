# Find given data point in data and update it.
def update_data(data, _p):
    for idx in range(len(data)):
        if data[idx][0] == _p[0] and data[idx][1] == _p[1]:
            data[idx] = _p


# Set union for joining neighbourhood 2 to neighbourhood 1   
def join(hood1, hood2):
    # For each point in hood2, append it to hood1 if it does not already exist.
    for p in hood2:
        found = False
        for _p in hood1:
            if p[0] == _p[0] and p[1] == _p[1]:
                found = True
                break
        if not found:
            hood1.append(p)
    return hood1


# Apply DBSCAN on data with given parameters
def dbscan(data, eps, min_pts):
    cluster = 0
    # For each data point in data
    for p in data:
        if not p[3]:
            p[3] = True # Mark as visited
            neighbor_pts = region_query(data, p, eps)
            if len(neighbor_pts) < min_pts:
                p[2] = -1  # Mark as noise
            else:  # is core
                cluster = cluster + 1  # next cluster
                expand_cluster(data, p, neighbor_pts, cluster, eps, min_pts)  # expand from core


def expand_cluster(data, p, neighbor_pts, cluster, eps, min_pts):
    p[2] = cluster  # Add p to cluster
    # For each neighbour of p
    for _p in neighbor_pts:
        if not _p[3]:
            _p[3] = True  # Mark as visited
            # Update in data itself
            _neighbor_pts = region_query(data, _p, eps)  # Get neighbours and add to current neighbours
            if len(_neighbor_pts) >= min_pts:  # if _p is core add its neighbours
                neighbor_pts = join(neighbor_pts, _neighbor_pts)
        if _p[2] is None or _p[2] == -1:
            _p[2] = cluster  # Add to cluster


def region_query(data, p, eps):
    neighborhood = []
    for _p in data:
        if distance(p, _p) <= eps**2:
            neighborhood.append(_p)  
    return neighborhood


# Squared Euclidean distance
def distance(_p, p):
    return (_p[0] - p[0]) ** 2 + (_p[1] - p[1]) ** 2
