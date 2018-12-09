from AdaCost import AdaCostClassifier
from xgboost import XGBClassifier
from imblearn.ensemble import RUSBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from load_dataset import get_features, get_labels
import numpy as np
import pickle
import os
from feature import *
import pandas as pd
import shutil


if __name__ == "__main__":
    # extract features/labels from data
    train_features, test_features = get_features()
    train_labels = get_labels()
    print(len(test_features))

    # prepare training data
    if "train_data.pkl" in os.listdir("./"):
        total_X, total_Y = pickle.load(open("train_data.pkl", "rb"))
    else:
        total_X = None
        total_Y = np.array([])
        print("pic info got.")
        build_cnt = 0
        for filename in sorted(train_labels.keys()):
            total_X = np.vstack([total_X, train_features[filename]]) if total_X is not None else train_features[filename]
            total_Y = np.append(total_Y, train_labels[filename])
            build_cnt += 1
            if build_cnt % 100 == 0:
                print("build dataset finish: %d/%d" % (build_cnt, len(sorted(train_labels.keys()))))
        pickle.dump((total_X, total_Y), open("train_data.pkl", "wb"))

    # prepare test data
    testset_X = None
    if "test_data.pkl" in os.listdir("./"):
        testset_X = pickle.load(open("test_data.pkl", "rb"))
    else:
        build_cnt = 0
        for filename in sorted(test_features.keys()):
            testset_X = np.vstack([testset_X, test_features[filename]]) if testset_X is not None else test_features[filename]
            build_cnt += 1
            if build_cnt % 100 == 0:
                print("build dataset finish: %d/%d" % (build_cnt, len(sorted(test_features.keys()))))
        pickle.dump(testset_X, open("test_data.pkl", "wb"))

    print(total_X.shape, testset_X.shape)

    # dimension reduction
    total_X, testset_X = PCA_reduce(total_X, testset_X)
    print(total_X.shape, testset_X.shape)

    # cross validation to choose a better model
    for train_idx, test_idx in KFold(n_splits=5).split(total_X):
        train_X, train_Y, test_X, test_Y = total_X[train_idx], total_Y[train_idx], total_X[test_idx], total_Y[test_idx]
        print("#Outlier:", np.sum(test_Y == 1))
        print("#Normal:", np.sum(test_Y == -1))

        # base_model = XGBClassifier()
        base_model = LogisticRegression(class_weight="balanced", solver='lbfgs')
        # base_model = SVC(probability=True, gamma='scale')
        model = base_model
        # model = RUSBoostClassifier(base_estimator=base_model, n_estimators=50)
        # model = AdaCostClassifier(base_estimator=base_model, n_estimators=50)
        print("Start training model...")
        model.fit(train_X, train_Y)
        y_pre = model.predict(test_X)

        print("[CV] Classification Report:\n", classification_report(test_Y, y_pre))
        print("[CV] F1-score:\n", f1_score(y_pre, test_Y))
        print("------------true")
        print("pred|\n    |\n    |\n    |\n    |\n")
        print("[CV] Confusion Matrix:\n", confusion_matrix(y_pre, test_Y))

    # outlier detection
    base_model = LogisticRegression(class_weight="balanced", solver='lbfgs')
    model = AdaCostClassifier(base_estimator=base_model, n_estimators=50)
    model.fit(total_X, total_Y)

    y_pre = model.predict(testset_X)
    res = pd.DataFrame()
    res["filename"] = list(test_features.keys())
    res["y_hat"] = y_pre
    all_outliers = res.mask(res["y_hat"] == -1).dropna()

    output_pd = res.copy()
    output_pd["ID"] = output_pd["filename"].apply(lambda x: x.split(".")[0])
    output_pd["Result"] = output_pd["y_hat"].apply(lambda x: 0 if x == -1 else 1)
    output_pd = output_pd.drop(columns=["y_hat", "filename"])
    output_pd.to_csv("Q1_output.csv", index=False)
    print(all_outliers)
    # all_outliers.to_csv("outlier_filelist.csv")
    # os.mkdir("./hahaha/")
    # for idx in all_outliers.index:
    #     filename = all_outliers.loc[idx]["filename"]
    #     shutil.copy("./Data_Q1/test/"+filename, "./hahaha/"+filename)




