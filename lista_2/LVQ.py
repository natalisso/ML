import os
import sys
import random 
import math

PARENT_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.abspath(os.path.join(PARENT_PATH, os.pardir))
SEED = 2

random.seed(SEED)
sys.path.insert(1, ROOT_PATH)
from models.knn import euclidean_distance


def get_best_matching_unit(codebooks, test_instance):
	distances = list()
	for codebook in codebooks:
		dist = euclidean_distance(codebook, test_instance)
		distances.append((codebook, dist))
	distances.sort(key=lambda tup: tup[1])
	return distances[0][0]

def choose_codebooks(training_data, n_codebooks):
	n_records = len(training_data)
	n_features = len(training_data[0])

	training_data_0 = [instance for instance in training_data if instance[n_features - 1] == 0] 
	training_data_1 = [instance for instance in training_data if instance[n_features - 1] == 1] 
	training_data = training_data_0 + training_data_1

	n_records_0 = len(training_data_0)
	n_records_1 = len(training_data_1)
	codebooks_0 = random.sample(training_data_0, k=min(n_codebooks,n_records_0))
	codebooks_1 = random.sample(training_data_1, k=min(n_codebooks,n_records_1))
	codebooks = codebooks_0 + codebooks_1
	return codebooks

def train_codebooks(training_data, n_codebooks, lrate, epochs, version="1"):
	codebooks = choose_codebooks(training_data, n_codebooks)

	for epoch in range(epochs):
		rate = lrate * math.exp(-epoch / 200)
		print("Epoch %d: rate=%.3f" % (epoch, rate))
		for row in training_data:
			# 1-NN of row over codebooks
			bmu = get_best_matching_unit(codebooks, row)

			for i in range(len(row)-1):
				# If they are from different classes
				if bmu[-1] == row[-1]:
					bmu[i] += rate * (row[i] - bmu[i])
				else:
					bmu[i] -= rate * (row[i] - bmu[i])
	return codebooks
