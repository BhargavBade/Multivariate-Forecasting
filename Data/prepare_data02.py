# prepare_data.py
import os
import pandas as pd
import numpy as np
import math
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
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
        
        # Select only alternate samples (2, 4, 6, ...)
        data = data.iloc[1::2].reset_index(drop=True)
        
        #data = data.iloc[:int(0.5 * len(data))].reset_index(drop=True)

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

        
        # #Exclude No column from the dataframe. It does not carry any weight in the dataframe.
        # data = df_encoded.drop(['No'], axis=1)
               
        # Drop all unnecessary columns at once
        # columns_to_drop = ['No', 'year', 'month', 'day', 'hour', 'season', 
        #                    'DEWP', 'HUMI', 'PRES', 'TEMP', 'Iws', 'precipitation',
        #                    'Iprec', 'cbwd_NE', 'cbwd_NW', 'cbwd_SE']
        
        columns_to_drop = ['No', 'year', 'month', 'day', 'hour']
        
        data = df_encoded.drop(columns=columns_to_drop, axis=1)
        
        return data

    def split_and_standardize_data(self, data):
        # Split data
        train_set, remaining_set = train_test_split(data, test_size=0.3, shuffle=False, random_state=42)
        val_set, test_set = train_test_split(remaining_set, test_size=0.75, shuffle=False, random_state=42)

        # Standardization
        # scaler = StandardScaler()
        scaler = MinMaxScaler(feature_range=(0, 1))
        #exclude_cols = ['year', 'month', 'day', 'hour', 'season']
        pm_column = 'PM'
        #num_cols = train_set.select_dtypes(include=[np.number]).columns.difference(exclude_cols)
        num_cols = train_set.select_dtypes(include=[np.number]).columns
        pm_index = num_cols.get_loc(pm_column)  # Get the index of PM in the num_cols
        
        train_set[num_cols] = scaler.fit_transform(train_set[num_cols])
        val_set[num_cols] = scaler.transform(val_set[num_cols])
        test_set[num_cols] = scaler.transform(test_set[num_cols])
        
        #return train_set, val_set, test_set
        return train_set, val_set, test_set, scaler, pm_index

    def split_sequences(self, input_sequences, output_sequence, n_steps_in, n_steps_out):
        X, y, past_pm25, datetime_info = [], [], [], []
        for i in range(0, len(input_sequences), n_steps_out):  # Step by n_steps_out  
            # find the end of the input, output sequence
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out - 1
            # check if we are beyond the dataset
            if out_end_ix >= len(input_sequences):
                break
            # gather input and output of the pattern
            seq_x = input_sequences.iloc[i:end_ix, :].values
            seq_y = output_sequence[end_ix-1:out_end_ix]
            # Extract PM2.5 values for the input sequence (aligned with the output)
            pm25_past_values = output_sequence[i:end_ix]  # PM2.5 values in the past window
            
            # Extract datetime information (first 4 columns) for the current input window
            datetime_window = input_sequences.iloc[end_ix-1:out_end_ix, :4].values  # Convert to numpy array
            
            # Append the sequences to their respective lists
            X.append(seq_x)
            y.append(seq_y)
            past_pm25.append(pm25_past_values)
            datetime_info.append(datetime_window)
            
        # Convert lists to numpy arrays for compatibility
        return np.array(X), np.array(y), np.array(past_pm25), np.array(datetime_info)

    
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


        # Define parameters for the split_sequences
        n_steps_in = params.n_steps_in  # Number of past time steps to use as input
        n_steps_out = params.n_steps_out # Number of future time steps to predict
    
        # Apply split_sequences function to create the dataset structure
        X_train, y_train, past_pm25_train, train_dt = self.split_sequences(train_data, train_labels, n_steps_in, n_steps_out)
        X_val, y_val, past_pm25_val, val_dt = self.split_sequences(val_data, val_labels, n_steps_in, n_steps_out)
        X_test, y_test, past_pm25_test, test_dt = self.split_sequences(test_data, test_labels, n_steps_in, n_steps_out)


        # Convert to tensors
        self.train_data_tensor = torch.Tensor(X_train).float()
        self.train_labels_tensor = torch.Tensor(y_train).float()
        self.past_pm25_train = torch.Tensor(past_pm25_train).float()
        self.train_pm_dt = torch.Tensor(train_dt).float()
    
        self.val_data_tensor = torch.Tensor(X_val).float()
        self.val_labels_tensor = torch.Tensor(y_val).float()
        self.past_pm25_val = torch.Tensor(past_pm25_val).float()
        self.val_pm_dt = torch.Tensor(val_dt).float()
    
    
        self.test_data_tensor = torch.Tensor(X_test).float()
        self.test_labels_tensor = torch.Tensor(y_test).float()
        self.past_pm25_test = torch.Tensor(past_pm25_test).float()
        self.test_pm_dt = torch.Tensor(test_dt).float()
        

        self.scaler = scaler
        self.pm_index = pm_index

    def get_pm_columns(self, data_frame):
        return [col for col in data_frame.columns if col.startswith('PM')]

    def get_main_features_columns(self, data_frame):
        return [col for col in data_frame.columns if not col.startswith('PM')]

    def get_tensors(self):
        return (self.train_data_tensor, self.train_labels_tensor,
                self.val_data_tensor, self.val_labels_tensor, self.past_pm25_val, self.val_pm_dt,
                self.test_data_tensor, self.test_labels_tensor, self.past_pm25_test, self.test_pm_dt,
                self.scaler, self.pm_index)

