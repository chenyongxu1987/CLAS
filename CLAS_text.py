import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import pandas as pd
import os
from scipy import sparse
from scipy.io import savemat

from itertools import *

import time
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from numpy import *
import numpy as np
from sklearn import metrics
from collections import Counter
from sklearn.preprocessing import LabelEncoder

import kagglehub
from datasets import load_dataset

from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords


def knn(dataset, k):
    '''
    Calculate the k neighbors of each point.
    Parameters
    -----------
    dataset: array-like, shape = [n_samples, n_features]
             n_samples is the number of points in the data set, and n_features is the dimension of the parameter space.
    k: int, the number of neighbor include the point.

    Return
    -------
    neigh_ind: array-like, the index of neighbor.
    neigh_dis: array-like, distance from neighbor point to target point.
    '''
    neigh_dis, neigh_ind = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute', n_jobs=-1).fit(
        dataset).kneighbors(dataset, return_distance=True)
    return neigh_dis, neigh_ind


def density(neigh_dis, k):
    '''
    The function of first step of the algorithm. Calculate the density of each point.
    Parameters
    -----------
    neigh_dis: array-like or sparse matrix, shape (n_samples, the number of k top nearest neighbors)
              The k top nearest neighbors matrix.
    k: The k top nearest neighbors.

    Returns
    -------
    density: The density of each point which is a list.
    '''
    kneis_dis = neigh_dis[:, k - 1]
    average_dis = np.mean(kneis_dis)

    final_radius = np.minimum(kneis_dis, average_dis)

    density = np.sum(neigh_dis[:, 1:] <= final_radius[:, None], axis=1)

    return density.tolist()


def MkNN(neigh_ind):
    '''
    The function of the second step of the algorithm. Find mutual neighbors of each point.

    Parameters
    ----------
    neigh_ind: List of lists
        The index of the neighbors of each point. Each sublist represents the neighbors of a point.

    Return
    -------
    Mneigh_ind: List of lists
        A list of mutual nearby indices for each point.
    '''
    N = len(neigh_ind)
    Mneigh_ind = [[x] for x in range(N)]

    neigh_dict = {v: set(neigh_ind[v]) for v in range(N)}

    for v in range(N):
        for point in neigh_ind[v]:
            if v != point and v in neigh_dict.get(point, set()):
                Mneigh_ind[v].append(point)

    return Mneigh_ind


def Mdensity(Mneigh_ind, density):
    '''
    Calculate the density of mutual neighbors of each point.
    Parameters
    -----------
    Mneigh_ind: list,shape (the index of k mutual nearest neighbors)
                The index of k mutual nearest neighbors.
    density: list, shape = [n_samples].

    Returns
    --------
    Mdensity: list, shape = [n_samples].
    '''
    Mdensity = []
    for p_ind in Mneigh_ind:
        nei_density_list = [density[j] for j in p_ind]
        Mdensity.append(nei_density_list)
    return Mdensity


def clusters_backbone(Mdensity, Mneigh_ind):
    '''
    The function of 3rd step of the algorithm. This is the backbone of cluster formation process.

    Parameters
    -----------
    Mdensity: list, shape (n_samples, the density of k mutual nearest neighbors)
              The density of k mutual nearest neighbors matrix.
    Mneigh_ind: list, shape (the index of k mutual nearest neighbors)
                The index of k mutual nearest neighbors.

    Returns
    --------
    ini_cluster : The initial cluster result which is a list.
    '''

    # Step 1: Identify initial potential clusters based on density
    clusters = []
    for i in range(len(Mdensity)):
        if len(Mdensity[i]) > 1 and Mdensity[i][0] >= max(Mdensity[i]):
            clusters.append(set(Mneigh_ind[i]))

    if not clusters:
        return []

    # Step 2: Using a dictionary to track cluster memberships
    # Use a dict to keep track of which elements belong to which cluster
    cluster_map = defaultdict(set)
    for i, cluster in enumerate(clusters):
        for elem in cluster:
            cluster_map[elem].add(i)  # For each element, store its cluster index

    # Step 3: Create a union-find (disjoint-set) structure for merging clusters
    parent = list(range(len(clusters)))  # Parent of each cluster
    rank = [0] * len(clusters)  # Rank for union by rank

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]

    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            if rank[rootX] > rank[rootY]:
                parent[rootY] = rootX
            elif rank[rootX] < rank[rootY]:
                parent[rootX] = rootY
            else:
                parent[rootY] = rootX
                rank[rootX] += 1

    # Step 4: Merge clusters with common neighbors (optimized with vectorization)
    # Instead of looping over all pairs, track clusters that share elements
    for elem, cluster_indices in cluster_map.items():
        cluster_indices = list(cluster_indices)  # Convert to list for easier manipulation
        if len(cluster_indices) > 1:
            # Union all clusters that share this element
            base_cluster = cluster_indices[0]
            for i in range(1, len(cluster_indices)):
                union(base_cluster, cluster_indices[i])

    # Step 5: Aggregate merged clusters
    final_clusters = defaultdict(set)
    for i in range(len(clusters)):
        root = find(i)
        final_clusters[root].update(clusters[i])

    # Step 6: Return the final clusters after merging
    return [list(cluster) for cluster in final_clusters.values()]


def assign(dataset, density, neigh_ind, ini_cluster):
    '''
    The function of 4th step of the algorithm.
    This is the process of forming the final cluster, assign an unmarked point to a marked point that is closest to it and has a higher density than it.
    Parameters
    -----------
    dataset: array-like, shape = [n_samples, n_features]
    density: The density of each point which is a list.
    neigh_ind: The index of the neighbors of each point.
    ini_cluster: Clustering result which is a list.

    Returns
    --------
    final_cluster: The final cluster result which is a list.
    plabel: The label of the final cluster.
    '''

    plabel = -np.ones(dataset.shape[0], dtype=int)
    for i, cluster in enumerate(ini_cluster):
        for point in cluster:
            plabel[point] = i

    low = np.where(plabel == -1)[0]
    high = np.where(plabel != -1)[0]

    if low.size == 0:
        return ini_cluster, plabel

    # Step 1: Find a marked point with a higher density than each unmarked point.
    low_density = np.array(density)[low]
    high_density = np.array(density)[high]

    mask = high_density >= low_density[:, np.newaxis]
    low_to_high = {low_index: high[mask[i]] for i, low_index in enumerate(low)}

    # Step 2:  Align the indices of the unmarked point neighbors with the density.。
    lownei_dis = {low_index: [] for low_index in low}
    for i, low_index in enumerate(low):
        high_density_points = low_to_high[low_index]
        neigh_indices = neigh_ind[low_index]

        valid_neighbors = np.isin(neigh_indices, high_density_points)  # (k_neighbors,)
        neighbor_indices_in_low = np.where(valid_neighbors)[0]
        lownei_dis[low_index] = neighbor_indices_in_low.tolist()

    # Step 3: Assign the unmarked points to the nearest marked points whose density is greater than that of the unmarked points.
    final_cluster = [cluster.copy() for cluster in ini_cluster]
    for low_index in low:
        if lownei_dis[low_index]:
            nearest_high_density_point_index = min(lownei_dis[low_index])
            nearest_high_density_point = neigh_ind[low_index][nearest_high_density_point_index]
            final_cluster[plabel[nearest_high_density_point]].append(low_index)
            plabel[low_index] = plabel[nearest_high_density_point]

    return final_cluster, plabel


def access(labels_true, labels_pred):
    '''
    Calculate ARI and NMI.
    Parameters
    -----------
    labels_true: list.
    labels_pred: list.

    Returns
    --------
    ARI: float, Adjusted Rand Index.
    NMI: float, Normalized Mutual Information.
    '''

    ARI = metrics.adjusted_rand_score(labels_true, labels_pred)
    NMI = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    return ARI, NMI


def cluster_with_parameters(dataset):
    neigh_dis_full, neigh_ind_full = knn(dataset, 100)
    f1 = open('results/reuters21578_results.txt', 'w')
    ARI_MAX = 0
    NMI_MAX = 0
    for k in range(2, 100):
        for l in range(2, 31):

            # dis
            start_time = time.time()
            # k_neigh = knn(dataset, k)

            k_neigh = (neigh_dis_full[:, :k], neigh_ind_full[:, :k])

            # neigh_dis, neigh_ind = knn(dataset, dataset.shape[0])
            # K_neigh = knn(dataset, k)
            K_neigh = (neigh_dis_full[:, :l], neigh_ind_full[:, :l])

            end_time = time.time()
            # print('dis,run_time:', end_time - start_time)

            # MKNN
            start_time0 = time.time()
            Mneigh_ind = MkNN(K_neigh[1])
            end_time0 = time.time()
            # print('MKNN,run_time0:', end_time0 - start_time0)

            # density
            start_time1 = time.time()
            den = density(k_neigh[0], k)
            end_time1 = time.time()

            # Mdensity
            start_time2 = time.time()
            Mden = Mdensity(Mneigh_ind, den)
            end_time2 = time.time()

            # clusters_backbone
            start_time3 = time.time()
            ini_cluster = clusters_backbone(Mden, Mneigh_ind)
            end_time3 = time.time()

            # assign
            start_time4 = time.time()
            final_cluster = assign(dataset, den, neigh_ind_full, ini_cluster)
            end_time4 = time.time()

            # Total
            run_time = end_time - start_time + end_time0 - start_time0 + end_time1 - start_time1 + end_time2 - start_time2 + end_time3 - start_time3 + end_time4 - start_time4

            # print('density,run_time1:', end_time1 - start_time1)
            # print('Mdensity,run_time2:', end_time2 - start_time2)
            # print('clusters_backbone,run_time3:', end_time3 - start_time3)
            # print('assign,run_time4:', end_time4 - start_time4)
            print('total,run_time:', run_time)

            # print(final_cluster[1])
            # score1 = silhouette_score(X, final_cluster[1])
            ARI_NMI = access(labels_true, final_cluster[1])
            ARI = ARI_NMI[0]
            NMI = ARI_NMI[1]
            if ARI >= ARI_MAX:
                # if NMI >= NMI_MAX:
                ARI_MAX = ARI
                NMI_MAX = NMI
                print('k=', str(k), ', l=', str(l), '  ARI:', str(ARI_NMI[0]), '  NMI:', str(ARI_NMI[1]))
            cluster_num = len(Counter(final_cluster[1]))
            f1.write('k=' + str(k) + ', l=' + str(l) + '  ARI:' + str(ARI_NMI[0]) + '  NMI:' + str(
                ARI_NMI[1]) + '  cluster_num:' + str(cluster_num) + " total run_time:" + str(run_time) + '\n')
    f1.close()
    return final_cluster


if __name__ == '__main__':
    # 1. load Reuters-21578
    ds = load_dataset("ucirvine/reuters21578", "ModApte")
    #print(ds["train"].column_names)

    train = ds["train"]
    test = ds["test"]
    texts = train["text"] + test["text"]
    topics = train["topics"] + test["topics"]

    # 2. get labels
    single_idxs = [i for i, tlist in enumerate(topics) if len(tlist) == 1]
    texts = [texts[i] for i in single_idxs]
    labels = [topics[i][0] for i in single_idxs]

    # 3. Label encoding
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # 4. TF–IDF Vectorization
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1,2),
        stop_words="english"
    )
    X = vectorizer.fit_transform(texts)
    labels_true = labels
    """
       Save lables
       """
    le = LabelEncoder()
    y_num = le.fit_transform(labels_true)
    # with open('label/label_reuters21578.in', 'w') as f:
    #     for label in y_num:
    #         f.write(f"{label}\n")

    # print(X.shape)
    # print(np.array(labels_true))
    # print(len(Counter(labels_true)))

    n_samples, n_features = X.shape

    # The number of non-zero elements
    nnz = X.nnz

    # Total number of elements
    total = n_samples * n_features

    # Sparsity rate
    sparsity = 1 - nnz / total

    print(f"Total number of elements: {nnz}")
    print(f"Sparsity rate: {sparsity:.4f}  （ {sparsity * 100:.2f}% 是 0）")

    pre_cluster = cluster_with_parameters(X)




