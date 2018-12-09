import numpy as np
import pandas as pd

DATA_DIR = "./Data_Q4/"
n_centroid = 2
max_iter = 100
sum_l1_dists_threshold = 0.0001


def get_l1_distance(v1, v2):
    return np.sum(np.abs(v1 - v2))


def get_l2_SSE(v1, v2):
    return np.sum((v1 - v2) ** 2)


if __name__ == "__main__":
    data_df = np.array(pd.read_csv(DATA_DIR + "Q4_Data.csv"))
    n_data = data_df.shape[0]
    centroids = [np.array([0, 0, 0, 0, 0, 0]), np.array([1, 1, 1, 1, 1, 1])]
    # centroids = [data_df[np.random.randint(0, n_data)] for i in range(n_centroid)]

    iter_cnt = 0
    for iter_idx in range(max_iter):
        iter_cnt += 1
        member_idxes = [list() for i in range(n_centroid)]
        labels = []

        # expectation
        for i in range(n_data):
            dist_p2c = []
            for c_idx in range(n_centroid):
                dist_p2c.append((c_idx, get_l2_SSE(data_df[i], centroids[c_idx])))
            dist_p2c = sorted(dist_p2c, key=lambda x: x[1])
            nearest_cid = dist_p2c[0][0]
            member_idxes[nearest_cid].append(i)
            labels.append(nearest_cid)

        # maximization
        new_centroids = [np.array([]) for i in range(n_centroid)]
        for c_idx in range(n_centroid):
            members = data_df[member_idxes[c_idx]]
            new_centroids[c_idx] = np.mean(members, axis=0)

        # compute SSE
        dists = []
        for p_idx in range(n_data):
            dists.append(get_l2_SSE(data_df[p_idx], centroids[labels[p_idx]]))
        SSE_p2c = np.sum(dists)
        if iter_idx < 2:
            print("Centers:", new_centroids)
            print("SSE:", SSE_p2c)

        # check stop criteria
        dists_centroid = []
        for c_idx in range(n_centroid):
            dists_centroid.append(get_l1_distance(new_centroids[c_idx], centroids[c_idx]))
        sum_l1_dist_centroids = np.sum(dists_centroid)
        centroids = new_centroids
        # print("sum_l1_dist:", sum_l1_dist_centroids)
        if sum_l1_dist_centroids < sum_l1_dists_threshold:
            break

    print("Total iteration num = ", iter_cnt)
    print("Final converged centers = ", centroids)

