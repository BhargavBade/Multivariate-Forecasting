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
import params_informer
importlib.reload(params_informer)


seq_len = params_informer.seq_len
label_len = params_informer.label_len
pred_len = params_informer.pred_len

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
        
       # Drop the data
       # Select only alternate samples (2, 4, 6, ...)
        # data = data.iloc[1::2].reset_index(drop=True)
        
        # # To select only a portion of the data
        # data = data.iloc[:int(0.2 * len(data))].reset_index(drop=True)

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
        
        # Compute the range of PM values
        pm_min = new_data['PM'].min()
        pm_max = new_data['PM'].max()
        pm_range = pm_max - pm_min
        pm_mean = new_data['PM'].mean()
        
        print(f"Range of PM values: {pm_range} (Min: {pm_min}, Max: {pm_max})")
        print(f"Avg of PM values: {pm_mean}")


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
        
        # columns_to_drop = ['No']
        
        df_encoded['datetime'] = pd.to_datetime(df_encoded[['year', 'month', 'day', 'hour']])
        
        columns_to_drop = ['No', 'year', 'month', 'day', 'hour', 'datetime']
        
        data = df_encoded.drop(columns=columns_to_drop, axis=1)
        date_stamp = df_encoded[['datetime']]
        
        return data, date_stamp
    
    
    
    def time_features(self, dates, freq='h'):

        # Extract time features using vectorized operations instead of apply()
        dates['month'] = dates.datetime.dt.month
        dates['day'] = dates.datetime.dt.day
        dates['weekday'] = dates.datetime.dt.weekday
        dates['hour'] = dates.datetime.dt.hour
        dates['minute'] = dates.datetime.dt.minute
        
        # Create 15-minute intervals for minute column (0, 1, 2, ..., 3 for each quarter of an hour)
        dates['minute'] = dates['minute'] // 15
    
        freq_map = {
            'y':[],'m':['month'],'w':['month'],'d':['month','day','weekday'],
            'b':['month','day','weekday'],'h':['month','day','weekday','hour'],
            't':['month','day','weekday','hour','minute'],
        }
        return dates[freq_map[freq.lower()]].values

    def split_and_standardize_data(self, data):
        # Split data
        train_set, remaining_set = train_test_split(data, test_size=0.3, shuffle=False, random_state=42)
        # val_set, test_set = train_test_split(remaining_set, test_size=0.75, shuffle=False, random_state=42)
        test_set = remaining_set
        val_set = remaining_set

        # Standardization
        # scaler = StandardScaler()
        scaler = MinMaxScaler(feature_range=(0, 1))
        #exclude_cols = ['year', 'month', 'day', 'hour', 'season']
        # exclude_cols = ['datetime']
        pm_column = 'PM'
        # num_cols = train_set.select_dtypes(include=[np.number]).columns.difference(exclude_cols)
        num_cols = train_set.select_dtypes(include=[np.number]).columns
        pm_index = num_cols.get_loc(pm_column)  # Get the index of PM in the num_cols
        
        train_set[num_cols] = scaler.fit_transform(train_set[num_cols])
        # val_set[num_cols] = scaler.transform(val_set[num_cols])
        test_set[num_cols] = scaler.transform(test_set[num_cols])
        val_set[num_cols] = test_set[num_cols] #Duplicate val set
        
        #return train_set, val_set, test_set
        return train_set, val_set, test_set, scaler, pm_index


    def split_sequences_for_informer(self, input_sequences, output_sequence, seq_len, date_stamp, label_len, pred_len):
        """
    Splits input_sequences and output_sequence into sequences suitable for Informer,
    including decoder inputs with padding for prediction.
    
    Args:
        input_sequences (DataFrame): Data containing all input features.
        output_sequence (Series): Data containing PM2.5 values to predict.
        n_steps_in (int): Number of time steps in encoder input.
        date_stamp : Date and time information.
        label_len (int): Length of the known label window used by the decoder.
        pred_len (int): Number of steps to predict.
        
    Returns:
        encoder_x, encoder_mark, decoder_x, decoder_mark, output_y (numpy arrays)
    """
    
        # Precompute the time features once for all date stamps
        self.freq = 'h'
        data_stamp = self.time_features(date_stamp)  # Precompute the time features for all rows

        encoder_x, decoder_x, output_y, encoder_mark, decoder_mark = [], [], [], [], []
        past_pm_data = []
        # for i in range(len(input_sequences) - seq_len - pred_len + 1):
        for i in range(0, len(input_sequences) - seq_len - pred_len, pred_len): 
            
            # Define start and end indices
            end_idx = i + seq_len  # Encoder end
            pred_end_idx = end_idx + pred_len  # Prediction end
            
            # Ensure we have enough data to construct both input and output
            if pred_end_idx > len(output_sequence):
                break
            
            # Encoder input
            seq_x = input_sequences.iloc[i:end_idx, :].values
            
            # Target output
            seq_y = output_sequence.iloc[end_idx:pred_end_idx].values
            
            # Extract PM2.5 values for the input sequence (aligned with the output)
            pm25_past_values = output_sequence[i:end_idx+1].values  # PM2.5 values in the past window
            # pm25_past_values = output_sequence.iloc[i + seq_len:i + seq_len + pred_len].values #duplicate pm2.5
            
            # Decoder input: known labels (past PM2.5) followed by zeros for prediction
            dec_inp = np.zeros((label_len + pred_len, 1))  # Assuming predicting PM2.5 (1 output feature)
            dec_inp[:label_len] = output_sequence.iloc[end_idx - label_len:end_idx].values.reshape(-1, 1)
            
           # TimeStamp data for the encoder and decoder
            enc_time = data_stamp[i:end_idx]
            dec_time = data_stamp[end_idx - label_len:pred_end_idx]
            #--------------------------------------------------------------------------------
            
            # Store the processed sequences
            encoder_x.append(seq_x)
            decoder_x.append(dec_inp)
            output_y.append(seq_y)
            encoder_mark.append(enc_time)
            decoder_mark.append(dec_time)
            past_pm_data.append(pm25_past_values)
        
        # Convert to numpy arrays for Informer
        return (np.array(encoder_x), np.array(encoder_mark), 
                np.array(decoder_x), np.array(decoder_mark),              
                np.array(output_y), np.array(past_pm_data))
    
    
    def prepare_data(self):
        # Load and clean data
        cities_data_path_list = os.listdir(self.data_dir)
        sample_data_path = os.path.join(self.data_dir, cities_data_path_list[0])
        raw_data = self.load_data(sample_data_path)
        processed_data, date_stamp = self.clean_and_process_data(raw_data)

        # Split and standardize
        train_set, val_set, test_set, scaler, pm_index = self.split_and_standardize_data(processed_data)

        # Separate labels
        train_data = train_set.drop("PM", axis=1)
        train_labels = train_set["PM"]

        val_data = val_set.drop("PM", axis=1)
        val_labels = val_set["PM"]

        test_data = test_set.drop("PM", axis=1)
        test_labels = test_set["PM"]

        #Splitting "date_stamp" for train,test and val.
        # Use the same indices as the train/test split to split date_stamp
        # Train split
        train_indices = train_set.index
        train_datestamp = date_stamp.loc[train_indices]
    
        # Validation split
        val_indices = val_set.index
        val_datestamp = date_stamp.loc[val_indices]
    
        # Test split
        test_indices = test_set.index
        test_datestamp = date_stamp.loc[test_indices] 
        
               
        # Apply split_sequences function to create the dataset structure
        X_train, X_tr_datestamp, y_train, y_tr_datestamp, output_y_train, past_pm_train = self.split_sequences_for_informer(train_data, train_labels, seq_len, train_datestamp, label_len, pred_len)
        # X_val, X_val_datestamp, y_val, y_val_datestamp, output_y_val, past_pm_val  = self.split_sequences_for_informer(val_data, val_labels, seq_len, val_datestamp, label_len, pred_len)
        X_test, X_test_datestamp, y_test, y_test_datestamp, output_y_test, past_pm_test = self.split_sequences_for_informer(test_data, test_labels, seq_len, test_datestamp, label_len, pred_len)
        
        # Duplicate val tensors
        X_val, X_val_datestamp, y_val, y_val_datestamp, output_y_val, past_pm_val  = self.split_sequences_for_informer(test_data, test_labels, seq_len, test_datestamp, label_len, pred_len)

        # Convert to tensors
        self.train_data_tensor = torch.Tensor(X_train).float()
        self.train_labels_tensor = torch.Tensor(y_train).float()
        self.train_mark_tns = torch.Tensor(X_tr_datestamp).float()
        self.train_labl_mark_tns = torch.Tensor(y_tr_datestamp).float()
        self.train_output_gt = torch.Tensor(output_y_train).float()
        self.train_past_pm = torch.Tensor(past_pm_train).float()

        self.val_data_tensor = torch.Tensor(X_val).float()
        self.val_labels_tensor = torch.Tensor(y_val).float()
        self.val_mark_tns = torch.Tensor(X_val_datestamp).float()
        self.val_labl_mark_tns = torch.Tensor(y_val_datestamp).float()
        self.val_output_gt = torch.Tensor(output_y_val).float()
        self.val_past_pm = torch.Tensor(past_pm_val).float()
        
        self.test_data_tensor = torch.Tensor(X_test).float()
        self.test_labels_tensor = torch.Tensor(y_test).float()
        self.test_mark_tns = torch.Tensor(X_test_datestamp).float()
        self.test_labl_mark_tns = torch.Tensor(y_test_datestamp).float()
        self.test_output_gt = torch.Tensor(output_y_test).float()
        self.test_past_pm = torch.Tensor(past_pm_test).float()

        self.scaler = scaler
        self.pm_index = pm_index

    def get_pm_columns(self, data_frame):
        return [col for col in data_frame.columns if col.startswith('PM')]

    def get_main_features_columns(self, data_frame):
        return [col for col in data_frame.columns if not col.startswith('PM')]

    def get_tensors(self):
        return (self.train_data_tensor, self.train_labels_tensor, self.train_mark_tns, self.train_labl_mark_tns, self.train_output_gt, self.train_past_pm,
                self.val_data_tensor, self.val_labels_tensor, self.val_mark_tns, self.val_labl_mark_tns, self.val_output_gt, self.val_past_pm,
                self.test_data_tensor, self.test_labels_tensor, self.test_mark_tns, self.test_labl_mark_tns, self.test_output_gt, self.test_past_pm,
                self.scaler, self.pm_index)


