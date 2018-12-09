from xgboost import XGBClassifier
from imblearn.ensemble import RUSBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import numpy as np
import pickle
import os
from feature import *
import pandas as pd
import shutil

if __name__ == "__main__":

    if "train_vms.pkl" not in os.listdir("./"):
        os.system("python feature.py")
        train_vms = pkl.load(open("train_vms.pkl", "rb"))
    else:
        train_vms = pkl.load(open("train_vms.pkl", "rb"))
    if "test_vms.pkl" not in os.listdir("./"):
        os.system("python feature.py")
        test_vms = pkl.load(open("test_vms.pkl", "rb"))
    else:
        test_vms = pkl.load(open("test_vms.pkl", "rb"))

    train_features = []
    train_labels = []
    for vms in train_vms:
        train_features += vms.feature_vectors
        train_labels += ([vms.label] * vms.n_time_slots)
    train_features = np.vstack(train_features)
    train_labels = np.vstack(train_labels)

    pca_processor = PCA(n_components=80)
    total_X = pca_processor.fit_transform(train_features)

    test_features = []
    for vms in test_vms:
        test_features += vms.feature_vectors
    test_features = np.vstack(test_features)
    test_X = pca_processor.transform(test_features)






