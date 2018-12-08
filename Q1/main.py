from AdaCost import AdaCostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from load_dataset import get_features, get_labels
import numpy as np


if __name__ == "__main__":
    train_features, test_features = get_features()
    train_labels = get_labels()
    total_X = None
    total_Y = np.array([])
    for filename in sorted(train_labels.keys()):
        total_X = np.vstack([total_X, train_features[filename]]) if total_X is not None else train_features[filename]
        total_Y = np.append(total_Y, train_labels[filename])
    for train_idx, test_idx in KFold(n_splits=5).split(total_X):
        train_X, train_Y, test_X, test_Y = total_X[train_idx], total_Y[train_idx], total_X[test_idx], total_Y[test_idx]
        base_model = XGBClassifier()
        model = AdaCostClassifier(base_estimator=base_model, n_estimators=20)
        model.fit(train_X, train_Y)
        y_pre = model.predict(test_X)
        print("[CV] Classification Report:\n", classification_report(y_pre, test_Y))
        print("[CV] F1-score:\n", f1_score(y_pre, test_Y))
        print("[CV] Confusion Matrix:\n", confusion_matrix(y_pre, test_Y))




