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

def is_inside_window(d_i, d_j, w):
	s = (1 - w) / (1 + w)

	if min((d_i / d_j), (d_j / d_i)) > s:
		return True
	return False

def get_best_matching_unit(codebooks, test_instance, version):
	distances = list()
	for codebook in codebooks:
		dist = euclidean_distance(codebook, test_instance)
		if dist != 0.0:
			distances.append((codebook, dist))
	distances.sort(key=lambda tup: tup[1])
	if version != "1":
		return [distances[0], distances[1]]
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

def train_codebooks(training_data, n_codebooks, lrate, epochs, version="1", w=0, epsilon=0):
	codebooks = choose_codebooks(training_data, n_codebooks)

	for epoch in range(epochs):
		rate = lrate * math.exp(-epoch / 200)
		print("Epoch %d: rate=%.3f" % (epoch, rate))
		for row in training_data:
			# Get the 1-NN or 2-NN from row over codebooks
			bmu = get_best_matching_unit(codebooks, row, version)

			for i in range(len(row)-1):
				if version == "1":
					# If they are not from different classes
					if bmu[-1] == row[-1]:
						bmu[i] += rate * (row[i] - bmu[i])
					else:
						bmu[i] -= rate * (row[i] - bmu[i])
				else:
					if is_inside_window(bmu[0][1], bmu[1][1], w):
						# If the 2-NN elements are from different classes
						if bmu[0][0][-1] != bmu[1][0][-1]:
							if bmu[0][0][-1] == row[-1]:
								bmu[0][0][i] += rate * (row[i] - bmu[0][0][i])
								bmu[1][0][i] -= rate * (row[i] - bmu[1][0][i])
							else:
								bmu[0][0][i] -= rate * (row[i] - bmu[0][0][i])
								bmu[1][0][i] += rate * (row[i] - bmu[1][0][i])
						else:
							if version == "3":
								if bmu[0][0][-1] == row[-1]:
									bmu[0][0][i] += epsilon * rate * (row[i] - bmu[0][0][i])
									bmu[1][0][i] += epsilon * rate * (row[i] - bmu[1][0][i])
	return codebooks
