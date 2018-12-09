import numpy as np
import scipy.sparse as sparse
from sklearn.decomposition import PCA


# reduce the dimension using PCA
def PCA_reduce(train_data, test_data):
    pca_analyser = PCA(n_components=80)
    reduced_data = pca_analyser.fit_transform(np.vstack([train_data, test_data]))
    print(reduced_data.shape)
    print("explained proportion:", pca_analyser.explained_variance_ratio_, "total:", np.sum(pca_analyser.explained_variance_ratio_))
    return reduced_data[:train_data.shape[0], :], reduced_data[train_data.shape[0]:, :]
