import os
import pandas as pd
import numpy as np
import time 
import sys
from LVQ import train_codebooks

PARENT_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.abspath(os.path.join(PARENT_PATH, os.pardir))
DATASETS_DIR =  os.path.join(ROOT_PATH, "datasets")
RESULTS_DIR = os.path.join(PARENT_PATH, "results/LVQ1")
SEED = 2

sys.path.insert(1, ROOT_PATH)

from models.knn import k_nearest_neighbors
from utils.metrics import get_accuracy, confusion_matrix
from utils.cross_validation import cross_validation_split
from utils.data_processing import data_pre_processing, targets_pre_processing


datasets = ["DATATRIEVE_transition", "KC2"]
n_folds = 5
learn_rate = 0.2
n_epochs = 1000
all_n_codebooks = [5, 10, 15]
all_n_neighbors = [1, 3]


for dataset_id, dataset_name in enumerate(datasets):
    results_filename = os.path.join(RESULTS_DIR, f"results_{dataset_name}.txt")
    results_file = open(results_filename, "w+")

    for n_neighbors in all_n_neighbors:
        results_file.write("Number of Neighbors: " + str(n_neighbors) + "\n")

        for n_codebooks in all_n_codebooks:
            # Getting the dataset and its infos
            dataset_file = os.path.join(DATASETS_DIR, dataset_name+".csv")
            dataset = pd.read_csv(dataset_file)             
            num_attributes = len(dataset.columns)
            # indices = [i for i in dataset.index]

            # Getting and processing the data and targets
            X = dataset.iloc[:, 0:num_attributes-1].to_numpy()
            y = dataset.iloc[:, num_attributes-1:]
            X = data_pre_processing(X)
            y = targets_pre_processing(y)

            # Gettig the k folds from all the data
            raw_data = [data + target for data, target in zip(X,y)]

            # Getting cookbooks vectors
            new_data = train_codebooks(raw_data, n_codebooks, learn_rate, n_epochs)

            # # Gettig the k folds from the new training data
            folds = cross_validation_split(new_data, n_folds)

            scores = []
            predictions = []
            targets = []

            for fold in folds:
                train_set = list(folds)
                train_set.remove(fold)
                train_set = sum(train_set, [])
                test_set = list()

                for row in fold:
                    row_copy = list(row)
                    test_set.append(row_copy)
                    row_copy[-1] = None
                
                predicted = k_nearest_neighbors(train_set, test_set, n_neighbors)
                actual = [row[-1] for row in fold]
                accuracy = get_accuracy(actual, predicted)
                scores.append(accuracy)
                predictions.extend(predicted)
                targets.extend(actual)

            total_accuracy = sum(scores)/float(len(scores))
            results_file.write("Accuracy (n_codebooks = {}) = {:.3f}%\n".format(n_codebooks, total_accuracy))
            results_file.write("y_actual = " + str(targets) + "\n")
            results_file.write("y_pred = " + str(predictions) + "\n\n")
            plot_subtitle = "Num Protótipos = " + str(n_codebooks) + ", Num Vizinhos mais próximos = " + str(n_neighbors)
            plot_file = os.path.join(RESULTS_DIR, f"plots/{dataset_name}_p={n_codebooks}_k={n_neighbors}.png")
            confusion_matrix(targets, predictions, dataset_name, plot_subtitle, plot_file)
            print('Mean Accuracy: %.3f%%' % (total_accuracy))