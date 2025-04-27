# data_loader.py
import pandas as pd
import numpy as np
import os # Included for consistency with user's plotting example structure

def load_feature_data(file1, file2, label1=0, label2=1):
    """
    Loads feature data from two CSV files, assigns labels, and concatenates.
    Assumes features are rows and trials are columns in the CSV, so transposes.
    """
    print(f"Loading feature data: {file1} (Label {label1}), {file2} (Label {label2})")
    try:
        # Trials are columns (120), features are rows (204) -> Transpose
        df1 = pd.read_csv(file1, header=None).T
        df2 = pd.read_csv(file2, header=None).T

        # Basic validation
        if df1.shape[1] != 204 or df2.shape[1] != 204:
             print(f"Warning: Expected 204 features per trial. Got {df1.shape[1]} & {df2.shape[1]}. Check data orientation.")

        y1 = np.full(df1.shape[0], label1) # Assign labels based on file
        y2 = np.full(df2.shape[0], label2)

        # Concatenate features (X) and labels (y)
        X = pd.concat([df1, df2], ignore_index=True).values
        y = np.concatenate([y1, y2])

        print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}.")
        return None, None
    except Exception as e:
        print(f"Error loading or processing feature data files: {e}")
        return None, None

def load_sensor_locations(filename="BCIsensor_xy.csv"):
    """Loads sensor XY locations from a CSV file."""
    print(f"Loading sensor locations from: {filename}")
    try:
        # Assumes two columns (x, y) and 102 rows (sensors)
        locations = pd.read_csv(filename, header=None, names=['x', 'y']).values

        # Basic validation
        if locations.shape[0] != 102 or locations.shape[1] != 2:
            print(f"Warning: Expected 102x2 shape for sensor locations, got {locations.shape}.")

        print(f"Sensor locations loaded. Shape: {locations.shape}")
        return locations[:, 0], locations[:, 1] # Return x and y columns separately
    except FileNotFoundError:
        print(f"Error: Sensor location file '{filename}' not found.")
        return None, None
    except Exception as e:
        print(f"Error loading sensor locations: {e}")
        return None, None