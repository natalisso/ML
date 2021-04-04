import os
import sys
import pandas as pd
from statistics import mean
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score

PARENT_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.abspath(os.path.join(PARENT_PATH, os.pardir))
DATASETS_DIR =  os.path.join(ROOT_PATH, "datasets")
RESULTS_DIR = os.path.join(PARENT_PATH, "results")
SEED = 2

sys.path.insert(1, ROOT_PATH)

from utils.metrics import get_evaluation_metrics
from utils.data_processing import data_pre_processing, targets_pre_processing, train_test_custom_split

datasets_params = {"DATATRIEVE_transition": 0.0864,
                   "KC2": 0.205,
                   "PC1": 0.0694,
                   "CM1": 0.0983
                  }

for dataset_name, minority_percentage in zip(datasets_params.keys(), datasets_params.values()):               
    results_dir = os.path.join(RESULTS_DIR, dataset_name)
    results_filename = os.path.join(results_dir, f"results_3_models.txt")


    # Get the raw dataset
    dataset_file = os.path.join(DATASETS_DIR, dataset_name+".csv")
    dataset = pd.read_csv(dataset_file) 
    num_attributes = len(dataset.columns)

    # Get and process the data and targets
    X = dataset.iloc[:, 0:num_attributes-1].to_numpy()
    y = dataset.iloc[:, num_attributes-1:]
    X = data_pre_processing(X)
    y = targets_pre_processing(y)
    data = [data + target for data, target in zip(X,y)]

    majority_target = 0 # normal -> without bugs
    minority_target = 1 # anomalous -> with bugs

    with open(results_filename, "w+") as results_file:

        for train_percentage in [0.3, 0.4, 0.5]:
        
            train_X, test_X, train_y, test_y = train_test_custom_split(data, train_percentage, majority_target)

            # Define the models
            models = [OneClassSVM(kernel='rbf', gamma=0.001, nu=0.25).fit(train_X)
                    # ,OneClassSVM(kernel='rbf', gamma=0.001, nu=0.5).fit(train_X)
                    ,OneClassSVM(kernel='rbf', gamma=0.001, nu=0.95).fit(train_X)
                    ,IsolationForest(contamination=minority_percentage, random_state=SEED).fit(train_X)
                    ]

            # Make predictions
            models_predictions = [[pred1,pred2,pred3] for pred1,pred2,pred3 in 
                                    zip(models[0].predict(test_X),models[1].predict(test_X),models[2].predict(test_X))]

            predictions = []
            for prediction in models_predictions:
                prediction = [1 if pred == -1 else 0 for pred in prediction]
                pred = mean(prediction)

                if abs(pred - 0) <= abs(pred - 1):
                    predictions.append(0)
                else:
                    predictions.append(1)

            # Evaluate the model
            TP = FN = FP = TN = 0
            for i in range(len(test_y)):
                if test_y[i] == 1 and predictions[i] == 1:
                    TP = TP+1
                elif test_y[i] == 1 and predictions[i] == 0:
                    FN = FN+1
                elif test_y[i] == 0 and predictions[i] == 1:
                    FP = FP+1
                else:
                    TN = TN +1

            accuracy = (TP+TN)/(TP+FN+FP+TN)
            sensitivity = TP/(TP+FN)
            specificity = TN/(TN+FP)
            # print(TP, FN, FP, TN)
            # print(accuracy)
            # print(sensitivity)
            # print(specificity)


            evaluation_metrics = get_evaluation_metrics(test_y, predictions, pos_label=1)
            results_file.write('P = %.2f\n' % train_percentage)
            results_file.write('Accuracy: %.3f\n' % evaluation_metrics["accuracy"])
            results_file.write('Sensitivity (TPR): %.3f\n' % sensitivity)
            results_file.write('False Alarms (FPR): %.3f\n' % (1 - specificity))
            results_file.write('Miss Rate (FNR): %.3f\n' % (FN/(FN+TP)))
            results_file.write('Specificity (TNR): %.3f\n' % specificity)
            results_file.write('F1 Score: %.3f\n' % evaluation_metrics["f1_measure"])
            results_file.write('Recall: %.3f\n' % evaluation_metrics["recall"])
            results_file.write('Precision: %.3f\n' % evaluation_metrics["precision"])
            results_file.write('TP: %d,  FN: %d,  FP: %d,  TN: %d\n\n' % (TP,FN,FP,TN))
