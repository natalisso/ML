import os
import pandas as pd
import numpy as np
from utils import(
    data_pre_processing,
    targets_pre_processing,
    k_fold_cross_validation,
    split_data,
    k_nearest_neighbors,
    get_accuracy
)


PARENT_PATH = os.path.dirname(__file__)
DATASETS_DIR =  os.path.join(PARENT_PATH, "datasets")
RESULTS_DIR = os.path.join(PARENT_PATH, "results")

datasets = ["DATATRIEVE_transition", "KC2"]
modes = ["knn", "weighted_knn", "adaptive_knn"]

for mode in modes:
    results_file = os.path.join(RESULTS_DIR, f"results_{mode}.txt")

    for dataset_id, dataset_name in enumerate(datasets):
        if dataset_id == 0:
            logger_mode = "w+"
        else:
            logger_mode = "a"

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

        with open(results_file, logger_mode) as logger:
            logger.write(f"Dataset: {dataset_name}\n")

            scores = []
            n_neighbors = [1,2,3,5,7,9,11,13,15]
            for k in n_neighbors:
                for i in range(num_folds):
                    # Generating the training and test data and targets
                    training_data, X_test, y_test = split_data(k_folds, X, y)
                
                    # Training and Testing
                    preditions = k_nearest_neighbors(training_data, X_test, k, mode)
                    accuracy = get_accuracy(y_test, preditions)
                    scores.append(accuracy)
                logger.write('Accuracy (k = {}): {:.3f}%\n'.format(k, sum(scores)/float(len(scores))))
            logger.write("\n")