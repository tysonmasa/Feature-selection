import random
import math
#import validator
import numpy as np

dataset = "small-test-dataset.txt"  

class Instance:
    def __init__(self, label, features):
        self.label = label
        self.features = features

# list of training instances
'''
trainInstances = []  
with open(dataset, "r") as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip()  
        line = line.split()  
        # change to float
        instanceLabel = float(line[0])  
        instanceFeatures = [float(x) for x in line[1:]]
        tempInstance = Instance(instanceLabel, instanceFeatures)
        trainInstances.append(tempInstance)
        '''

class Classifier:
    def __init__(self):
        self.trainingSet = []

    def train(self, x, y):
        trainInstances = []
        for i in x:
            tempInstance = Instance(y[i], x[i])
            trainInstances.append(tempInstance)
        self.trainingSet = trainInstances
        #print("Training set loaded with", len(self.trainingSet), "instances")

    def test(self, testInstance):
        nearest_instance = None
        nearest_distance = float('inf')

        for trainInstance in self.trainingSet:
            distance = self.euclidean_distance(testInstance.features, trainInstance.features)
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
            X_test, y_test = np.array([X_subset[i]]), np.array([y[i]])
            self.classifier.train(X_train, y_train)
            y_pred = self.classifier.test(X_test)
            correct += np.count_nonzero(y_pred == y_test)
        return correct / len(X_subset) * 100



def random_evaluation(features):
    return random.random()

def forward_selection(num_features):
    current_set_of_features = []
    best_overall_accuracy = 0
    best_feature_set = []

    for i in range(num_features):
        best_feature = None
        best_accuracy = 0
        
        for feature in range(1, num_features + 1):
            if feature not in current_set_of_features:
                accuracy = random_evaluation(current_set_of_features + [feature])
                print(f"Using feature(s) {current_set_of_features + [feature]} accuracy is {accuracy * 100:.1f}%")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_feature = feature
        
        if best_feature is not None:
            current_set_of_features.append(best_feature)
            if best_accuracy > best_overall_accuracy:
                best_overall_accuracy = best_accuracy
                best_feature_set = current_set_of_features.copy()
            print(f"Best set of features so far: {current_set_of_features}, with accuracy {best_accuracy * 100:.1f}%")

    print(f"Finished search!! The best feature subset is {best_feature_set}, which has an accuracy of {best_overall_accuracy * 100:.1f}%")
    return best_feature_set, best_overall_accuracy

def backward_elimination(num_features):
    current_set_of_features = list(range(1, num_features + 1))
    best_overall_accuracy = random_evaluation(current_set_of_features)
    best_feature_set = current_set_of_features.copy()

    print(f"Initial set {current_set_of_features} accuracy is {best_overall_accuracy * 100:.1f}%")
    
    for i in range(num_features - 1):
        worst_feature = None
        best_accuracy = 0
        
        for feature in current_set_of_features:
            temp_set = current_set_of_features.copy()
            temp_set.remove(feature)
            accuracy = random_evaluation(temp_set)
            print(f"Using feature(s) {temp_set} accuracy is {accuracy * 100:.1f}%")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                worst_feature = feature
        
        if worst_feature is not None:
            current_set_of_features.remove(worst_feature)
            if best_accuracy > best_overall_accuracy:
                best_overall_accuracy = best_accuracy
                best_feature_set = current_set_of_features.copy()
            print(f"Best set of features so far: {current_set_of_features}, with accuracy {best_accuracy * 100:.1f}%")

    print(f"Finished search!! The best feature subset is {best_feature_set}, which has an accuracy of {best_overall_accuracy * 100:.1f}%")
    return best_feature_set, best_overall_accuracy

def main():
    num_features = int(input("Please enter the total number of features: "))
    print("Type the number of the algorithm you want to run:")
    print("1. Forward Selection")
    print("2. Backward Elimination")
    algorithm_choice = int(input())

    '''
    classifier = Classifier()
    # Train the classifier with the training instances
    classifier.train(trainInstances)  
    '''

    if algorithm_choice == 1:
        print("Forward Selection Trace:")
        forward_selection_trace = forward_selection(num_features)
    elif algorithm_choice == 2:
        print("Backward Elimination Trace:")
        backward_elimination_trace = backward_elimination(num_features)
    else:
        print("Invalid choice. Please select 1 or 2.")

    '''
    # Allow the user to input a test instance
    print("Please enter the test instance feature values separated by spaces:")
    test_features = list(map(float, input().split()))
    test_instance = Instance(0, test_features)
    predicted_label = classifier.test(test_instance)
    print("Predicted label for the test instance:", predicted_label)
    '''

    smalldata = np.loadtxt('small-test-dataset.txt')
    largedata = np.loadtxt('large-test-dataset.txt')

    small_X = smalldata[:, 1:] #feats
    small_y = smalldata[:, 0] #labels
    large_X = largedata[:, 1:]
    large_y = largedata[:, 0]

    classifier = Classifier()
    validator = Validator(classifier)

    accuracy = validator.evaluate(small_X, small_y, [2, 4, 6])
    print(f"smalldata Accuracy: {accuracy}%")

    accuracy = validator.evaluate(large_X, large_y, [0, 14, 26])
    print(f"largedata Accuracy: {accuracy}%")

main()
