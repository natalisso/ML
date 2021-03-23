import math 
import numpy as np

def euclidean_distance(instance_1, instance_2):
	distance = 0.0
	num_features = len(instance_1) - 1

	for feature in range(num_features):
		distance += (instance_1[feature] - instance_2[feature])**2

	return math.sqrt(distance)


def get_neighbors(training_data, test_instance, n_neighbors):
	distances = list()
	
	for train_row in training_data:
		dist = euclidean_distance(test_instance, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(n_neighbors):
		neighbors.append(distances[i][0])

	return neighbors


def predict_classification(training_data, test_instance, n_neighbors):
	neighbors = get_neighbors(training_data, test_instance, n_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)

	return prediction


def k_nearest_neighbors(training_data, X_test, n_neighbors):
	predictions = list()
	
	for test_instance in X_test:
		output = predict_classification(training_data, test_instance, n_neighbors)
		predictions.append(output)
		
	return predictions
