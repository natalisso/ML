import os
import sys
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

PARENT_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.abspath(os.path.join(PARENT_PATH, os.pardir))
DATASETS_DIR =  os.path.join(ROOT_PATH, "datasets")
RESULTS_DIR = os.path.join(PARENT_PATH, "results")
SEED = 2

sys.path.insert(1, ROOT_PATH)

from utils.metrics import get_evaluation_metrics
from utils.data_processing import data_pre_processing, targets_pre_processing, train_test_custom_split

# Get the raw dataset
dataset_name = "PC1"
dataset_file = os.path.join(DATASETS_DIR, dataset_name+".csv")
dataset = pd.read_csv(dataset_file) 
num_attributes = len(dataset.columns)

# Get and process the data and targets
X = dataset.iloc[:, 0:num_attributes-1].to_numpy()
y = dataset.iloc[:, num_attributes-1:]
X = data_pre_processing(X)
y = targets_pre_processing(y)
data = [data + target for data, target in zip(X,y)]

# print(y)
majority = len(dataset[dataset['defects'] == 0])
minority = len(dataset[dataset['defects'] == 1])

print(majority, minority)

majority_target = 0
minority_target = 1
minority_percentage = 0.0694
train_percentage = 0.5

train_X, test_X, train_y, test_y = train_test_custom_split(data, train_percentage, majority_target)

# Define the model
model = OneClassSVM(gamma='scale', nu=0.0846)
# model = EllipticEnvelope(contamination=0.0846, random_state=SEED)
# model = IsolationForest(contamination=minority_percentage, random_state=SEED)

# Train on majority class
model.fit(train_X)

# Test the model
predictions = model.predict(test_X)


# Transform inliers 1 and outliers -1 predictions into original targets
predictions = [majority_target if target == 1 else minority_target for target in predictions]

# Evaluate the model
score = f1_score(test_y, predictions, pos_label=minority_target)
print('F1 Score: %.3f' % score)
evaluation_metrics = get_evaluation_metrics(test_y, predictions, pos_label=minority_target)
print('F1 Score: %.3f' % evaluation_metrics["f1_measure"])
print('Recall: %.3f' % evaluation_metrics["recall"])
print('Precision: %.3f' % evaluation_metrics["precision"])
print(evaluation_metrics["true_positives"],evaluation_metrics["false_positives"],evaluation_metrics["true_negatives"],evaluation_metrics["false_negatives"])
