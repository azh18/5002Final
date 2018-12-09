# -*- coding:utf-8 -*-
import numpy as np
from sklearn.ensemble import AdaBoostClassifier

outlier_weight_reduce_factor = 0.08
outlier_weight_increase_factor = 8


class AdaCostClassifier(AdaBoostClassifier):
    def _boost_real(self, iboost, X, y, sample_weight, random_state):
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict_proba = estimator.predict_proba(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1),
                                       axis=0)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        # Construct y coding as described in Zhu et al [2]:
        #
        #    y_k = 1 if c == k else -1 / (K - 1)
        #
        # where K == n_classes_ and c, k in [0, K) are indices along the second
        # axis of the y coding with c being the index corresponding to the true
        # class label.
        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1. / (n_classes - 1), 1.])
        y_coding = y_codes.take(classes == y[:, np.newaxis])

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba = y_predict_proba  # alias for readability
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)

        # Boost weight using multi-class AdaBoost SAMME.R alg
        estimator_weight = (-1. * self.learning_rate
                            * ((n_classes - 1.) / n_classes)
                            * (y_coding * np.log(y_predict_proba)).sum(axis=1))

        # add different cost for imbalanced labels
        if not iboost == self.n_estimators - 1:
            sample_weight *= np.exp(estimator_weight *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)) *
                                    self._beta(y, y_predict))  # adjusted cost self._beta(y, y_predict)
        return sample_weight, 1., estimator_error

    def _beta(self, y, y_hat):
        res = []
        for i in zip(y, y_hat):
            # 1 = outlier, -1 = inlier
            if i[0] == i[1] and i[1] == 1:
                res.append(outlier_weight_reduce_factor)   # outlier -> weight reduce slower
            elif i[0] == i[1] and i[1] == -1:
                res.append(1) # inlier -> weight reduce normal
            elif i[0] == 1 and i[1] == -1:
                res.append(outlier_weight_increase_factor)  # outlier -> weight increase faster
            elif i[0] == -1 and i[1] == 1:
                res.append(1)  # inlier -> weight add normal
            else:
                print(i[0], i[1])

        return np.array(res)
