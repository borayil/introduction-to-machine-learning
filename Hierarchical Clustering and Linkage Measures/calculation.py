from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score


def single_linkage(data):
    return linkage(data, method='single', metric='euclidean')


def average_linkage(data):
    return linkage(data, method='average', metric='euclidean')


def complete_linkage(data):
    return linkage(data, method='complete', metric='euclidean')


def ward_linkage(data):
    return linkage(data, method='ward', metric='euclidean')


def clusters(data, amt):
    return fcluster(data, amt, criterion='maxclust')


def silhoutte_scores(data, fclusters):
    return silhouette_score(data, fclusters)
