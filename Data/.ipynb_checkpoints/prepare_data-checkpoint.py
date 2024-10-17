# prepare_data.py
import os
import pandas as pd
import numpy as np
import math
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
import importlib

sys.path.append('..')  # Add the parent directory to the Python path
import params
importlib.reload(params)

class DataPreparer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_data_tensor = None
        self.train_labels_tensor = None
        self.val_data_tensor = None
        self.val_labels_tensor = None
        self.test_data_tensor = None
        self.test_labels_tensor = None

    def load_data(self, data_path):
        return pd.read_csv(data_path)

    def clean_and_process_data(self, data):
        # Process your data here (same logic you already have)
        #MAIN_DIR_PATH = './01_PM2.5 Chinese Weather data
        MAIN_DIR_PATH = self.data_dir
        cities_data_path_list = os.listdir(MAIN_DIR_PATH)

        #Beijing city data analysis
        sample_data_path = os.path.join(MAIN_DIR_PATH, cities_data_path_list[0])
        print(sample_data_path)
        data = self.load_data(sample_data_path)

        # Create a single PM column and perform feature engineering
        pm_cols = self.get_pm_columns(data)
        features_columns = self.get_main_features_columns(data)
        features_columns.append('PM')
        features_columns = [col for col in features_columns if col != 'PM']

        new_data_rows_list = []
        for idx, (index, row) in enumerate(data.iterrows()):
            mn = row[pm_cols].mean()
            if not math.isnan(mn):
                temp_row = {col: row[col] for col in features_columns}
                temp_row['PM'] = mn
                new_data_rows_list.append(temp_row)

        new_data = pd.DataFrame(new_data_rows_list)

        # One-hot encode 'cbwd' and other steps
        df_encoded = pd.get_dummies(new_data, columns=['cbwd'], prefix='cbwd')
        df_encoded = df_encoded.fillna(0)
        df_encoded = df_encoded.drop(columns=['cbwd_cv'])

        #Convert "cbwd_" features into numericals.
        # Assuming 'df' is your DataFrame and you want to apply it to multiple columns
        columns_to_convert = ['cbwd_NE', 'cbwd_NW', 'cbwd_SE']  # List of columns that need conversion

        for col in columns_to_convert:
            df_encoded[col] = df_encoded[col].replace({'True': 1, 'False': 0}).astype(int)

        
        #Exclude No column from the dataframe. It does not carry any weight in the dataframe.
        data = df_encoded.drop(['No'], axis=1)
        
        return data

    def split_and_standardize_data(self, data):
        # Split data
        train_set, remaining_set = train_test_split(data, test_size=0.4, shuffle=False, random_state=42)
        val_set, test_set = train_test_split(remaining_set, test_size=0.75, shuffle=False, random_state=42)

        # Standardization
        scaler = StandardScaler()
        exclude_cols = ['year', 'month', 'day', 'hour', 'season']
        pm_column = 'PM'
        num_cols = train_set.select_dtypes(include=[np.number]).columns.difference(exclude_cols)
        pm_index = num_cols.get_loc(pm_column)  # Get the index of PM in the num_cols
        
        train_set[num_cols] = scaler.fit_transform(train_set[num_cols])
        val_set[num_cols] = scaler.transform(val_set[num_cols])
        test_set[num_cols] = scaler.transform(test_set[num_cols])

        #return train_set, val_set, test_set
        return train_set, val_set, test_set, scaler, pm_index

    def prepare_data(self):
        # Load and clean data
        cities_data_path_list = os.listdir(self.data_dir)
        sample_data_path = os.path.join(self.data_dir, cities_data_path_list[0])
        raw_data = self.load_data(sample_data_path)
        processed_data = self.clean_and_process_data(raw_data)

        # Split and standardize
        train_set, val_set, test_set, scaler, pm_index = self.split_and_standardize_data(processed_data)

        # Separate labels
        train_data = train_set.drop("PM", axis=1)
        train_labels = train_set["PM"]

        val_data = val_set.drop("PM", axis=1)
        val_labels = val_set["PM"]

        test_data = test_set.drop("PM", axis=1)
        test_labels = test_set["PM"]

        # Data windowing
        window_size = params.window_size
        stride = params.stride
        self.train_data_tensor, self.train_labels_tensor = self.sliding_window(train_data, train_labels, window_size, stride)
        self.val_data_tensor, self.val_labels_tensor = self.sliding_window(val_data, val_labels, window_size, stride)
        self.test_data_tensor, self.test_labels_tensor = self.sliding_window(test_data, test_labels, window_size, stride)

        # Convert all datasets to tensors
        self.train_data_tensor = torch.from_numpy(self.train_data_tensor).float()
        self.train_labels_tensor = torch.from_numpy(self.train_labels_tensor).float()

        self.val_data_tensor = torch.from_numpy(self.val_data_tensor).float()
        self.val_labels_tensor = torch.from_numpy(self.val_labels_tensor).float()

        self.test_data_tensor = torch.from_numpy(self.test_data_tensor).float()
        self.test_labels_tensor = torch.from_numpy(self.test_labels_tensor).float()

        self.scaler = scaler
        self.pm_index = pm_index

    def sliding_window(self, data, labels, window_size, stride):
        data_windows = []
        label_windows = []

        for i in range(0, len(data) - window_size + 1, stride):
            data_windows.append(data.iloc[i:i + window_size].values)
            label_windows.append(labels.iloc[i:i + window_size].values)

        return np.array(data_windows), np.array(label_windows)

    def get_pm_columns(self, data_frame):
        return [col for col in data_frame.columns if col.startswith('PM')]

    def get_main_features_columns(self, data_frame):
        return [col for col in data_frame.columns if not col.startswith('PM')]

    def get_tensors(self):
        return (self.train_data_tensor, self.train_labels_tensor,
                self.val_data_tensor, self.val_labels_tensor,
                self.test_data_tensor, self.test_labels_tensor,
                self.scaler, self.pm_index)
