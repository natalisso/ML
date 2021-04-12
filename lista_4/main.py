import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from statistics import mean, stdev
from clustering import generate_kstar_clusters

PARENT_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.abspath(os.path.join(PARENT_PATH, os.pardir))
DATASETS_DIR =  os.path.join(ROOT_PATH, "datasets")
RESULTS_DIR = os.path.join(PARENT_PATH, "results")
SEED = 2

sys.path.insert(1, ROOT_PATH)
from utils.data_processing import data_pre_processing

datasets = ["DATATRIEVE_transition", "CM1"]
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
    # Initializing results file        
    results_dir = os.path.join(RESULTS_DIR, dataset_name)
    results_filename = os.path.join(results_dir, f"results_clustering.txt")
    with open(results_filename, 'w+') as results_file:
        pass

    # Get the features and labels already pre-processed
    X, y = get_dataset(dataset_name)

    # Get the k folds using stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=5)
    folds = skf.split(X, y)

    # Train and test the data on each folder
    with open(results_filename, 'a') as results_file:
        n_fold = 0
        results = {"1-NN": {"f1": [], "tpr": [], "fpr": []},
                   "NB": {"f1": [], "tpr": [], "fpr": []},
                   "NB Clustered": {"f1": [], "tpr": [], "fpr": []}
                  }

        for train_index, test_index in folds:
            results_file.write("Fold %d:\n" % (n_fold+1))
            X_train = np.array(X)[train_index.astype(int)]
            X_test = np.array(X)[test_index.astype(int)]
            y_train, y_test = y[train_index], y[test_index]
            results_file.write(f"y_test = {y_test}\n\n")

            # 1-NN
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(X_train, y_train)
            knn_preds = knn.predict(X_test)
            knn_results = classification_report(y_test, knn_preds, target_names=target_names)
            results_file.write(f" 1-NN:\n{knn_results}\n") 
            results_file.write(f"knn_preds = {knn_preds}\n")
            results["1-NN"]["f1"].append(f1_score(y_test, knn_preds))   
            tn, fp, fn, tp = confusion_matrix(y_test, knn_preds).ravel()
            results["1-NN"]["tpr"].append(tp / (tp + fn))    
            results["1-NN"]["fpr"].append(fp / (fp + tn))   

            # Naive Bayes
            nb = GaussianNB()
            nb.fit(X_train, y_train)
            nb_preds = nb.predict(X_test)
            nb_results = classification_report(y_test, nb_preds, target_names=target_names)
            results_file.write(f" Naive Bayes:\n{nb_results}\n")
            results_file.write(f"nb_preds = {nb_preds}\n")
            results["NB"]["f1"].append(f1_score(y_test, nb_preds))
            tn, fp, fn, tp = confusion_matrix(y_test, nb_preds).ravel()
            results["NB"]["tpr"].append(tp / (tp + fn))    
            results["NB"]["fpr"].append(fp / (fp + tn))
        
            # Naive Bayes (with clustering)
            results_file.write(f" Naive Bayes (with clustering):\n")
            old_k_star = 0
            y_train_final = []
            X_train_final = []
            map_old_new_labels = [[],[]]

            # Select and generate the best k clusters for each class in the training set
            for i in range(2):
                features = [row for row,label in zip(X_train, y_train) if label == i]
                k_star, sse, silhouette_coefficient, y_train_new = generate_kstar_clusters(features, i, dataset_name, n_fold)
                
                X_train_final.extend(features)
    
                map_old_new_labels[i] = y_train_new + old_k_star
                y_train_final.extend(map_old_new_labels[i])
                old_k_star = k_star

                results_file.write("  * Class %d: k_star = %d, sse = %f, silhouette coef = %f\n" % (i, k_star, sse, silhouette_coefficient))

            # Classification with the Naive Bayes algorithm and new labels
            nb_clustered = GaussianNB()
            X_train = np.array(X_train_final)
            y_train_clustered = np.array(y_train_final)
            nb_clustered.fit(X_train, y_train_clustered)
            nb_clustered_preds = nb_clustered.predict(X_test)
            mapped_nb_preds = [0 if pred in set(map_old_new_labels[0]) else 1 for pred in nb_clustered_preds]
            results_file.write(f"nb_clustered_preds = {mapped_nb_preds}\n")
            nb_clustered_results = classification_report(y_test, mapped_nb_preds, target_names=target_names)
            results_file.write(f" {nb_clustered_results}\n\n")
            results["NB Clustered"]["f1"].append(f1_score(y_test, mapped_nb_preds))
            tn, fp, fn, tp = confusion_matrix(y_test, mapped_nb_preds).ravel()
            results["NB Clustered"]["tpr"].append(tp / (tp + fn))    
            results["NB Clustered"]["fpr"].append(fp / (fp + tn))
            
            n_fold += 1

        # Final results (mean +- confidence interval)    
        results_file.write(f"FINAL SCORES:\n")
        for model, result in results.items():
            f1_mean = mean(result["f1"])
            f1_confidence_interval = 2 * stdev(result["f1"])
            tpr_mean = mean(result["tpr"])
            tpr_confidence_interval = 2 * stdev(result["tpr"])
            fpr_mean = mean(result["fpr"])
            fpr_confidence_interval = 2 * stdev(result["fpr"])
            results_file.write(f" {model}:\n * F1 Score = {f1_mean} +- {f1_confidence_interval}\n * TPR = {tpr_mean} +- {tpr_confidence_interval}\n * FPR = {fpr_mean} +- {fpr_confidence_interval}\n")

       