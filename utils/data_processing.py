import pandas as pd
from sklearn.preprocessing import StandardScaler


def data_pre_processing(data):
    # Ajusting the scale of the data attributes
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data).tolist()
    return scaled_data

def targets_pre_processing(targets):
    # Transforming str target to int
    categorical_to_int = lambda x: 1 if (x == 'yes' or x == 1) else 0
    targets_enc = targets.applymap(categorical_to_int).to_numpy().tolist()
    return targets_enc
