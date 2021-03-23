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
	# Separating the training data and targets
	training_indices = [j for j in k_folds[0]]
	training_data = [(data[i],targets[i]) for i in training_indices]

	# Separating the test data from their target attribute
	test_indices = [j for j in k_folds[1]]
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

def get_accuracy(y_test, preditions):
	correct = 0
	
	for i in range(len(y_test)):
		if y_test[i] == preditions[i]:
			correct += 1
	return correct / float(len(y_test)) * 100.0
