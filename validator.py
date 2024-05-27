import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

class Validator:
    def __init__(self, classifier: BaseEstimator):
        self.classifier = classifier

    def evaluate(self, X: np.ndarray, y: np.ndarray, feature_subset: list) -> float:
        X_subset = X[:, feature_subset]
        accuracy = []
        for i in range(len(X_subset)):
            X_train = np.concatenate((X_subset[:i], X_subset[i+1:]))
            y_train = np.concatenate((y[:i], y[i+1:]))
            X_test, y_test = np.array([X_subset[i]]), np.array([y[i]])
            self.classifier.fit(X_train, y_train)
            y_pred = self.classifier.predict(X_test)
            accuracy.append(accuracy_score(y_test, y_pred))
        return np.mean(accuracy)
    