import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, classification_report
from clustering import generate_kstar_clusters

PARENT_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.abspath(os.path.join(PARENT_PATH, os.pardir))
DATASETS_DIR =  os.path.join(ROOT_PATH, "datasets")
RESULTS_DIR = os.path.join(PARENT_PATH, "results")
SEED = 2

sys.path.insert(1, ROOT_PATH)
from utils.data_processing import data_pre_processing

datasets = ["PC1", "CM1"]
target_names = ['class 0', 'class 1']


def get_dataset(dataset_name):
    # Get the raw dataset
    dataset_file = os.path.join(DATASETS_DIR, dataset_name+".csv")
    dataset = pd.read_csv(dataset_file) 

    # Split and process the raw dataset into features (X) and labels (y)
    X = dataset.iloc[:, 0:-1].to_numpy()
    X = data_pre_processing(X)
    y = dataset.iloc[:,-1].squeeze().map(lambda x: 1 if (x == 'yes' or x == 1 or x == '1' or x == 'true') else 0).to_numpy()

    return X, y


for dataset_name in datasets:     
    print(dataset_name)  
    # Initializing results file        
    results_dir = os.path.join(RESULTS_DIR, dataset_name)
    results_filename = os.path.join(results_dir, f"results_clustering.txt")
    with open(results_filename, 'w+') as results_file:
        pass

    # Get the features and labels already pre-processed
    X, y = get_dataset(dataset_name)

    # Get the k folds using stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
    folds = skf.split(X, y)

    # Train and test the data on each folder
    with open(results_filename, 'a') as results_file:
        n_fold = 0
        results = {"1-NN":[], "NB": [], "NB Clustered": []}
        for train_index, test_index in folds:
            results_file.write("Fold %d:\n" % (n_fold+1))
            X_train = np.array(X)[train_index.astype(int)]
            X_test = np.array(X)[test_index.astype(int)]
            y_train, y_test = y[train_index], y[test_index]

            # 1-NN
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(X_train, y_train)
            knn_preds = knn.predict(X_test)
            knn_results = classification_report(y_test, knn_preds, target_names=target_names)
            results_file.write(f" 1-NN:\n{knn_results}\n")
            results["1-NN"].append()
            

            # Naive Bayes
            nb = GaussianNB()
            nb.fit(X_train, y_train)
            nb_preds = nb.predict(X_test)
            nb_results = classification_report(y_test, nb_preds, target_names=target_names)
            results_file.write(f" Naive Bayes:\n{nb_results}\n")
        

            # Naive Bayes (with clustering)
            results_file.write(f" Naive Bayes (with clustering):\n")
            old_k_star = 0
            y_train_final = []
            map_old_new_labels = [[],[]]

            # Select and generate the best k clusters for each class in the training set
            for i in range(2):
                features = [row for row,label in zip(X_train, y_train) if label == i]
                k_star, y_train_new = generate_kstar_clusters(features, i, dataset_name, n_fold)

                map_old_new_labels[i] = y_train_new + old_k_star
                y_train_final.extend(map_old_new_labels[i])
                old_k_star = k_star

                results_file.write("  * Class %d: k_star = %d\n" % (i, k_star))

            nb_clustered = GaussianNB()
            y_train_clustered = np.array(y_train_final)
            nb_clustered.fit(X_train, y_train_clustered)
            nb_clustered_preds = nb_clustered.predict(X_test)
            mapped_nb_preds = [0 if pred in map_old_new_labels[0] else 1 for pred in nb_clustered_preds]
            results_file.write(f"nb_preds = {nb_clustered_preds}\nmapped_nb_preds = {mapped_nb_preds}\ny_test = {y_test}\n")
            nb_clustered_results = classification_report(y_test, mapped_nb_preds, target_names=target_names)
            results_file.write(f" {nb_clustered_results}\n\n")
            
            n_fold += 1