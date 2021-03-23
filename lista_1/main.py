import os
import pandas as pd
import numpy as np
import time 
import sys

from knn import k_nearest_neighbors
from utils import(
    data_pre_processing,
    targets_pre_processing,
    k_fold_cross_validation,
    split_data,
    get_accuracy
)


PARENT_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.abspath(os.path.join(PARENT_PATH, os.pardir))
DATASETS_DIR =  os.path.join(ROOT_PATH, "datasets")
RESULTS_DIR = os.path.join(PARENT_PATH, "results")


datasets = ["DATATRIEVE_transition", "KC2"]
modes = ["knn", "weighted_knn", "adaptive_knn"]

for mode in modes:
    results_file = os.path.join(RESULTS_DIR, f"results_{mode}.txt")

    for dataset_id, dataset_name in enumerate(datasets):
        logger_mode = "a"
        if dataset_id == 0:
            logger_mode = "w+"

        with open(results_file, logger_mode) as logger:
            print(f"DATASET: {dataset_name}\n")
            logger.write(f"Dataset: {dataset_name}\n")

            # Getting the dataset and its infos
            dataset_file = os.path.join(DATASETS_DIR, dataset_name+".csv")
            dataset = pd.read_csv(dataset_file)             
            num_attributes = len(dataset.columns)
            indices = [i for i in dataset.index]  

            # Pre-processing the data and targets
            X = dataset.iloc[:, 0:num_attributes-1].to_numpy()
            X = data_pre_processing(X)
            y = dataset.iloc[:, num_attributes-1:]
            y = targets_pre_processing(y)
        
            # Generating the k-folds
            num_folds = 10
            k_folds = k_fold_cross_validation(num_folds, indices) 

            y_acctual = []
            y_pred = []
            scores = []
            n_neighbors = [1,2,3,5,7,9,11,13,15]
            for k in n_neighbors:
                print("k =", k)
                start_time = time.time()
                for i in range(num_folds):
                    # Generating the training and test data and targets
                    training_data, X_test, y_test = split_data(k_folds[i], X, y)
  
                    # Training and Testing
                    preditions = k_nearest_neighbors(training_data, X_test, k, mode)
  
                    y_acctual.extend(y_test)
                    y_pred.extend(preditions)
                    accuracy = get_accuracy(y_test, preditions)
                    scores.append(accuracy)
                execution_time = time.time() - start_time
                logger.write("Execution Time (k = {}): {} seconds\n".format(k, execution_time))
                total_accuracy = sum(scores)/float(len(scores))
                logger.write("Accuracy (k = {}): {:.3f}%\n".format(k, total_accuracy))
                print("Execution Time (k = {}): {} seconds".format(k, execution_time))
                print("Accuracy (k = {}): {:.3f}%\n".format(k, total_accuracy))
            logger.write("y_actual = "+str(y_acctual)+"\n")
            logger.write("y_pred = "+str(y_pred)+"\n\n")
            print()
            
