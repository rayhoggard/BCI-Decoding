import pandas as pd
import numpy as np
import os

def load_feature_data(file1, file2, label1=0, label2=1):
# Loads feature data
# Trials are columns (120), features are rows (204) so transpose
    df1 = pd.read_csv(file1, header=None).T
    df2 = pd.read_csv(file2, header=None).T
    y1 = np.full(df1.shape[0], label1)
    y2 = np.full(df2.shape[0], label2)
    X = pd.concat([df1, df2], ignore_index=True).values
    y = np.concatenate([y1, y2])
    print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}")
    return X, y

def load_sensor_locations(filename="BCIsensor_xy.csv"):
# Loads sensor data
    locations = pd.read_csv(filename, header=None, names=['x', 'y']).values
    if locations.shape[0] != 102 or locations.shape[1] != 2:
        print(f"Warning: Expected 102x2 shape, got {locations.shape}.")
    print(f"Sensor locations loaded. Shape: {locations.shape}")
    return locations[:, 0], locations[:, 1]