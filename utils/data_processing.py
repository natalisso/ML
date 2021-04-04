import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer
import math
import random
from random import seed

SEED = 2 


def data_pre_processing(data):
    # Ajusting the scale of the data attributes
    scaler = PowerTransformer()
    scaled_data = scaler.fit_transform(data).tolist()
    return scaled_data

def targets_pre_processing(targets):
    # Transforming str target to int
    categorical_to_int = lambda x: 1 if (x == 'yes' or x == 1 or x == '1' or x == 'true') else 0
    targets_enc = targets.applymap(categorical_to_int).to_numpy().tolist()
    return targets_enc

def train_test_custom_split(data, train_percentage, majority_class):
    seed(SEED)

    trainable_set = [row for row in data if row[-1] == majority_class]
    n_samples = len(trainable_set)
    n_train_samples = math.floor(n_samples * train_percentage)
    train_set = random.sample(trainable_set, n_train_samples)
    test_set = [row for row in data if row not in train_set]

    train_X, train_y = [row[:-1] for row in train_set], [row[-1] for row in train_set]
    test_X, test_y = [row[:-1] for row in test_set], [row[-1] for row in test_set]

    return train_X, test_X, train_y, test_y

