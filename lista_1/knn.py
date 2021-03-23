import math 
import numpy as np

def euclidean_distance(instance_1, instance_2):
	distance = 0.0
	num_attributes = len(instance_1)

	for attribute in range(num_attributes):
		distance += (instance_1[attribute] - instance_2[attribute])**2

	return math.sqrt(distance)

def get_min_radius(training_data, current_instance, current_instance_id):
	epsilon = 1e-14
	min_radius = 0.0
	
	all_radius = [(euclidean_distance(current_instance, instance[0]) - epsilon) if instance_id != current_instance_id else np.inf for instance_id,instance in enumerate(training_data)]
	min_radius = min(all_radius)
	if min_radius <= 0.0:
		min_radius = epsilon
	return min_radius

def get_neighbors(training_data, test_instance, n_neighbors, mode):
	distances = list()
	
	for instance_id, training_instance in enumerate(training_data):
		distance = euclidean_distance(test_instance, training_instance[0])
		if mode == "adaptive_knn":
			min_radius = get_min_radius(training_data, training_instance[0], instance_id)
			distance = distance / min_radius
		distances.append((instance_id, distance))
	
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(n_neighbors):
		neighbors.append(distances[i])

	return neighbors

def predict_classification(training_data, test_instance, n_neighbors, mode):
	neighbors = get_neighbors(training_data, test_instance, n_neighbors, mode)
	output_values = [np.argmax(training_data[i[0]][1]) for i in neighbors]	# Getting the classes of the neighbors
	
	weights = None
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
