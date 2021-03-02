import random
import math 
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import (
    KNeighborsClassifier, 
    RadiusNeighborsClassifier
)
from sklearn.preprocessing import(
    StandardScaler,
    OneHotEncoder,
    LabelEncoder,
)

SEED = 2


def k_fold_cross_validation(k, indices):
    random.Random(SEED).shuffle(indices)    # Ramdomizing the samples indices
    dataset_size = len(indices)
    subset_size = round(dataset_size / k)   # Getting the samples subsets size
    subsets = [indices[x:x+subset_size] for x in range(0, dataset_size, subset_size)]  # Creating the samples subsets
    
    k_folds = []
    for num_fold in range(k):
        test = subsets[num_fold]  # Testing set of the i-th fold = the i-th subset
        train = []                # Training set of the i-th fold = all other subsets
        for subset in subsets:
            if subset != test:
                train.extend(subset)
        k_folds.append((train, test))

    return k_folds

def split_data(k_folds, data, targets):
	for i in range(len(k_folds)):
		# Separating the training data and targets
		training_indices = [j for j in k_folds[i][0]]
		training_data = [(data[i],targets[i])  for i in training_indices]

		# Separating the test data from their target attribute
		test_indices = [j for j in k_folds[i][1]]
		X_test = [data[i] for i in test_indices]
		y_test = [np.argmax(targets[i]) for i in test_indices]
	return training_data, X_test, y_test

def data_pre_processing(data):
	# Ajusting the scale of the data attributes
	scaler = StandardScaler()
	return scaler.fit_transform(data) 

def targets_pre_processing(targets):
	# Transforming targets in one hot enconding labels
	le = LabelEncoder()
	enc = OneHotEncoder()

	targets_le = targets.apply(le.fit_transform)
	enc.fit(targets_le)
	targets_enc = enc.transform(targets_le).toarray()
	return targets_enc

# def get_distance(mode, instance_1, instance_2):
# 	distance = euclidean_distance(instance_1, instance_2)	# Default uses uniform weights
# 	if mode == "weighted_knn":
# 		distance = weighted_euclidean_distance(instance_1, instance_2)	# Assigns weights proportional to the inverse of the distance from the query point
# 	return distance

def euclidean_distance(instance_1, instance_2):
	distance = 0.0
	num_attributes = len(instance_1)
	for attribute in range(num_attributes):
		distance += (instance_1[attribute] - instance_2[attribute])**2
	return math.sqrt(distance)

def get_neighbors(training_data, test_instance, n_neighbors):
	distances = list()
	for instance_id, training_instance in enumerate(training_data):
		distance = euclidean_distance(test_instance, training_instance[0])
		distances.append((instance_id, distance))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(n_neighbors):
		neighbors.append(distances[i])
	return neighbors

def predict_classification(training_data, test_instance, n_neighbors, mode):
	neighbors = get_neighbors(training_data, test_instance, n_neighbors)
	weights = None
	output_values = [np.argmax(training_data[i[0]][1]) for i in neighbors]	# Getting the classes of the neighbors
	if mode == "weighted_knn":
		weights = [(1 / (i[1])**2) if i[1] != 0 else 1 for i in neighbors]	# Weight = 1 / (d(xq,xi)^2)
	prediction = np.bincount(output_values,weights=weights).argmax()
	return prediction

def k_nearest_neighbors(training_data, X_test, n_neighbors, mode):
	predictions = list()
	for test_instance in X_test:
		output = predict_classification(training_data, test_instance, n_neighbors, mode)
		predictions.append(output)
	return predictions

def get_accuracy(y_test, preditions):
	correct = 0
	for i in range(len(y_test)):
		if y_test[i] == preditions[i]:
			correct += 1
	return correct / float(len(y_test)) * 100.0
