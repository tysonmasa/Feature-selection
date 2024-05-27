import numpy as np
import main
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
class Validator:
    def __init__(self, classifier: BaseEstimator):
        self.classifier = classifier

    def evaluate(self, X: np.ndarray, y: np.ndarray, feature_subset: list) -> float:
        X_subset = X[:, feature_subset]
        correct = 0
        for i in range(len(X_subset)):
            X_train = np.concatenate((X_subset[:i], X_subset[i+1:]))
            y_train = np.concatenate((y[:i], y[i+1:]))
            X_test, y_test = np.array([X_subset[i]]), np.array([y[i]])
            self.classifier.fit(X_train, y_train)
            y_pred = self.classifier.predict(X_test)
            correct += np.count_nonzero(y_pred == y_test)
        return correct / len(X_subset) * 100
    
smalldata = np.loadtxt('small-test-dataset.txt')
largedata = np.loadtxt('large-test-dataset.txt')

small_X = smalldata[:, 1:]
small_y = smalldata[:, 0]
large_X = largedata[:, 1:]
large_y = largedata[:, 0]

classifier = KNeighborsClassifier(2)
validator = Validator(classifier)

accuracy = validator.evaluate(small_X, small_y, [2, 4, 6])
print(f"smalldata Accuracy: {accuracy}%")

accuracy = validator.evaluate(large_X, large_y, [0, 14, 26])
print(f"largedata Accuracy: {accuracy}%")