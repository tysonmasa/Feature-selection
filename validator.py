import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Validator:
    def init(self, classifier: BaseEstimator):
        self.classifier = classifier

    def evaluate(self, X: np.ndarray, y: np.ndarray, feature_subset: list) -> float:
        X_subset = X[:, feature_subset]
        X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.3, random_state=42)
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy