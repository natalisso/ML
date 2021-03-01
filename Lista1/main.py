import os
import pandas as pd
import numpy as np
from utils import(
    k_fold_cross_validation,
    data_pre_processing,
    targets_pre_processing,
    split_data,
    euclidean_distance,
    train,
    test
)


PARENT_PATH = os.path.dirname(__file__)
DATASET_PATH =  os.path.join(PARENT_PATH, "datasets/DATATRIEVE_transition.csv")


# Getting the dataset and its infos
dataset = pd.read_csv(DATASET_PATH)             
num_attributes = len(dataset.columns)
indices = [i for i in dataset.index]  

# Pre-processing the data and targets
data = dataset.iloc[:, 0:num_attributes-1].to_numpy()
targets = dataset.iloc[:, num_attributes-1:]
data = data_pre_processing(data)
targets = targets_pre_processing(targets)

# Generating the k-folds
k = 10
k_folds = k_fold_cross_validation(k, indices)   

test_acc = []
for i in range(k):
    # Generating the training and test data and targets
    X_train, y_train, X_test, y_test = split_data(k_folds, data, targets)

    # Training the model
    model = train("knn", X_train, y_train)

    # Evaluating the model
    acc = test(model, X_test, y_test)
    test_acc.append(acc)

# Gathering the model evaluations
total_test_acc = np.mean(test_acc)
print(f"\nTest Accuracy = {total_test_acc}")
