import random
import math
import numpy as np
class Instance:
    def __init__(self, label, features):
        self.label = label
        self.features = features

class Classifier:
    def __init__(self):
        self.trainingSet = []

    def train(self, x, y):
        trainInstances = []
        for i in range(len(x)):
            tempInstance = Instance(y[i], x[i]) #(label,features)
            trainInstances.append(tempInstance)
        self.trainingSet = trainInstances
        #print("Training set loaded with", len(self.trainingSet), "instances")

    def test(self, testInstance):
        nearest_instance = None
        nearest_distance = float('inf')

        for trainInstance in self.trainingSet:
            distance = self.euclidean_distance(testInstance, trainInstance.features)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_instance = trainInstance
        
        return nearest_instance.label if nearest_instance else None

    @staticmethod
    def euclidean_distance(features1, features2):
        return math.sqrt(sum((f1 - f2) ** 2 for f1, f2 in zip(features1, features2)))

class Validator:
    def __init__(self, classifier):
        self.classifier = classifier

    def evaluate(self, X: np.ndarray, y: np.ndarray, feature_subset: list) -> float:
        X_subset = X[:, feature_subset]
        correct = 0
        for i in range(len(X_subset)):
            X_train = np.concatenate((X_subset[:i], X_subset[i+1:]))
            y_train = np.concatenate((y[:i], y[i+1:]))
            X_test, y_test = X_subset[i], np.array([y[i]])
            self.classifier.train(X_train, y_train)
            y_pred = self.classifier.test(X_test)
            correct += np.count_nonzero(y_pred == y_test)
        return correct / len(X_subset) * 100

def evaluation(features, x, y):
    classifier = Classifier()
    validator = Validator(classifier)
    accuracy = validator.evaluate(x, y, features)
    return accuracy

def forward_selection(num_features, x, y):
    current_set_of_features = []
    best_overall_accuracy = 0
    best_feature_set = []

    for i in range(num_features):
        best_feature = None
        best_accuracy = 0
        
        for feature in range(0, num_features):
            if feature not in current_set_of_features:
                accuracy = evaluation(current_set_of_features + [feature], x, y)
                print(f"Using feature(s) {current_set_of_features + [feature]} accuracy is {accuracy:.1f}%")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_feature = feature
        
        if best_feature is not None:
            current_set_of_features.append(best_feature)
            if best_accuracy > best_overall_accuracy:
                best_overall_accuracy = best_accuracy
                best_feature_set = current_set_of_features.copy()
            print(f"Best set of features so far: {current_set_of_features}, with accuracy {best_accuracy:.1f}%")

    best_feature_set = [feature + 1 for feature in best_feature_set]
    print(f"Finished search!! The best feature subset is {best_feature_set}, which has an accuracy of {best_overall_accuracy:.1f}%")
    return best_feature_set, best_overall_accuracy

def backward_elimination(num_features, x, y):
    current_set_of_features = list(range(0, num_features))
    best_overall_accuracy = evaluation(current_set_of_features, x, y)
    best_feature_set = current_set_of_features.copy()
    print(f"Initial set {current_set_of_features} accuracy is {best_overall_accuracy:.1f}%")
    
    for i in range(num_features - 1):
        worst_feature = None
        best_accuracy = 0
        
        for feature in current_set_of_features:
            temp_set = current_set_of_features.copy()
            temp_set.remove(feature)
            accuracy = evaluation(temp_set, x, y)
            temp_set = [temp + 1 for temp in temp_set]
            print(f"Using feature(s) {temp_set} accuracy is {accuracy:.1f}%")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                worst_feature = feature
        
        if worst_feature is not None:
            current_set_of_features.remove(worst_feature)
            if best_accuracy > best_overall_accuracy:
                best_overall_accuracy = best_accuracy
                best_feature_set = current_set_of_features.copy()
            print(f"Best set of features so far: {current_set_of_features}, with accuracy {best_accuracy:.1f}%")

    best_feature_set = [feature + 1 for feature in best_feature_set]
    print(f"Finished search!! The best feature subset is {best_feature_set}, which has an accuracy of {best_overall_accuracy:.1f}%")
    return best_feature_set, best_overall_accuracy

def main():
    print("Type the number of the dataset you want to use:")
    print("1. Part II Sample Dataset")
    print("2. Group Personal Dataset 51")
    datasetchoice = int(input())
    
    print("Type the number of the type of dataset you want to use:")
    print("1. Small Dataset")
    print("2. Large Dataset")
    datachoice = int(input())

    print("Type the number of the algorithm you want to run:")
    print("1. Forward Selection")
    print("2. Backward Elimination")
    algorithm_choice = int(input())

    if datasetchoice == 1:
        smalldata = np.loadtxt('small-test-dataset.txt')
        largedata = np.loadtxt('large-test-dataset.txt')
    else:
        smalldata = np.loadtxt('CS170_Spring_2024_Small_data__51.txt')
        largedata = np.loadtxt('CS170_Spring_2024_Large_data__51.txt')
    
    small_X = smalldata[:, 1:] #feats
    small_y = smalldata[:, 0] #labels
    large_X = largedata[:, 1:]
    large_y = largedata[:, 0]

    if algorithm_choice == 1:
        if datachoice == 1:
            print(f"Forward Selection Trace on small dataset with {len(smalldata[0,1:])} features:")
            forward_selection_trace = forward_selection(len(smalldata[0,1:]), small_X, small_y)
        elif datachoice == 2:
            print("Forward Selection Trace on large dataset:")
            forward_selection_trace = forward_selection(len(largedata[0,1:]), large_X, large_y)
    elif algorithm_choice == 2:
        if datachoice == 1:
            print("Backward Elimination Trace on small dataset:")
            backward_elimination_trace = backward_elimination(len(smalldata[0,1:]), small_X, small_y)
        elif datachoice == 2:
            print("Backward Elimination Trace on large dataset:")
            backward_elimination_trace = backward_elimination(len(largedata[0,1:]), large_X, large_y)
    else:
        print("Invalid choice. Please select 1 or 2.")

main()
