# LVQ for the Ionosphere Dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import os
import sys
import pandas as pd
import numpy as np


PARENT_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.abspath(os.path.join(PARENT_PATH, os.pardir))
DATASETS_DIR =  os.path.join(ROOT_PATH, "datasets")

sys.path.insert(1, ROOT_PATH)

from utils import(
    data_pre_processing
)
from models.knn import k_nearest_neighbors

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = []
    predictions = []
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
        predictions.extend(predicted)
    return scores, predictions

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Locate the best matching unit
def get_best_matching_unit(codebooks, test_row):
	distances = list()
	for codebook in codebooks:
		dist = euclidean_distance(codebook, test_row)
		distances.append((codebook, dist))
	distances.sort(key=lambda tup: tup[1])
	return distances[0][0]

# Make a prediction with codebook vectors
def predict(codebooks, test_row):
	bmu = get_best_matching_unit(codebooks, test_row)
	return bmu[-1]

# Create a random codebook vector
def random_codebook(train):
	n_records = len(train)
	n_features = len(train[0])
	codebook = [train[randrange(n_records)][i] for i in range(n_features)]
	return codebook

# Train a set of codebook vectors
def train_codebooks(train, n_codebooks, lrate, epochs):
	codebooks = [random_codebook(train) for i in range(n_codebooks)]
	for epoch in range(epochs):
		rate = lrate * (1.0-(epoch/float(epochs)))
		for row in train:
			bmu = get_best_matching_unit(codebooks, row)
			for i in range(len(row)-1):
				error = row[i] - bmu[i]
				if bmu[-1] == row[-1]:
					bmu[i] += rate * error
				else:
					bmu[i] -= rate * error
	return codebooks

# LVQ Algorithm
# def learning_vector_quantization(training_data, X_test, n_neighbors, n_codebooks, lrate, epochs):
#     codebooks = train_codebooks(training_data, n_codebooks, lrate, epochs)
#     print("Len test = %d" % (len(X_test)))
#     predictions = k_nearest_neighbors(codebooks, X_test, n_neighbors)

#     return predictions

def learning_vector_quantization(train, test, n_codebooks, lrate, epochs):
	codebooks = train_codebooks(train, n_codebooks, lrate, epochs)
	predictions = list()
	for row in test:
		output = predict(codebooks, row)
		predictions.append(output)
	return(predictions)

# Test LVQ on Ionosphere dataset
seed(1)
# load and prepare data
dataset_name = "KC2"
dataset_file = os.path.join(DATASETS_DIR, dataset_name+".csv")
dataset = pd.read_csv(dataset_file)             
num_attributes = len(dataset.columns)
indices = [i for i in dataset.index]

# Getting data and targets
X = dataset.iloc[:, 0:num_attributes-1].to_numpy()
X = data_pre_processing(X).tolist()
categorical_to_int = lambda x: 1 if (x == 'yes') else 0
y = dataset.iloc[:, num_attributes-1:].applymap(categorical_to_int).to_numpy().tolist()

training_data = [data + target for data, target in zip(X,y)]

# evaluate algorithm
n_folds = 5
n_neighbors = 1
learn_rate = 0.1
n_epochs = 50
n_codebooks = 10

# codebooks_vectors = train_codebooks(training_data, n_codebooks, learn_rate, n_epochs)
scores = evaluate_algorithm(training_data, learning_vector_quantization, n_folds, n_neighbors, n_codebooks, learn_rate, n_epochs)
print('Scores: %s' % scores)
print('Total num of Scores: %d' % len(scores))
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))