import random

dataset = "small-test-dataset.txt" # change to user input

class Instance:
    def __init__(self, label, features):
        self.label = label
        self.features = features

trainInstances = [] #list of training instances
with open(dataset, "r") as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip() #remove newline symbol
        line = line.split() #turn the line into a list
        instanceLabel = int(line[0])
        instanceFeatures = []
        for x in line[1:]:
            instanceFeatures.append(int(x))
        #Instance(label, feature(list))
        tempInstance = Instance(instanceLabel, instanceFeatures)
        trainInstances.append(tempInstance)


class Classifier:
    def __init__(self):
        self.trainingSet = []
    #input set of training instances
    def train(self, trainingSet):
        self.trainingSet = trainingSet
        print("set trained")
    #input a test instance, compare euc distance between test instance and all training points
    def test(testInstance):
        pass

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
    # Ask the user to enter the total number of features
    num_features = int(input("Please enter the total number of features: "))
    
    # Ask the user to select the algorithm to run
    print("Type the number of the algorithm you want to run:")
    print("1. Forward Selection")
    print("2. Backward Elimination")
    algorithm_choice = int(input())
    
    # Run the selected algorithm and display the trace
    if algorithm_choice == 1:
        print("Forward Selection Trace:")
        forward_selection_trace = forward_selection(num_features)
    elif algorithm_choice == 2:
        print("Backward Elimination Trace:")
        backward_elimination_trace = backward_elimination(num_features)
    else:
        print("Invalid choice. Please select 1 or 2.")

main()