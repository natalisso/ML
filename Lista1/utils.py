import random
import math 
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
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

def data_pre_processing(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data) 

def targets_pre_processing(targets):
    le = LabelEncoder()
    enc = OneHotEncoder()

    targets_le = targets.apply(le.fit_transform)
    enc.fit(targets_le)
    targets_enc = enc.transform(targets_le).toarray()
    return targets_enc

def split_data(k_folds, data, targets):
    for i in range(len(k_folds)):
        # Separating the training data from their target attribute
        training_indices = [j for j in k_folds[i][0]]
        X_train = [data[i] for i in training_indices]
        y_train = [targets[i] for i in training_indices]

        # Separating the test data from their target attribute
        test_indices = [j for j in k_folds[i][1]]
        X_test = [data[i] for i in test_indices]
        y_test = [targets[i] for i in test_indices]
    return X_train, y_train, X_test, y_test

def euclidean_distance(instance_1, instance_2):
    distance = 0
    num_attributes = len(instance_1)
    for attribute in range(num_attributes):
        distance += pow((instance_1[attribute] - instance_2[attribute]), 2)
    return math.sqrt(distance)

def train(model, X_train, y_train):
    if model == "knn": 
        model = KNeighborsClassifier(n_neighbors=1)
    return model.fit(X_train, y_train)

def test(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)