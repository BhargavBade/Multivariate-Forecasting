# prepare_data.py
import os
import pandas as pd
import numpy as np
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

data_ft = params_informer.enc_inp

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
        
        MAIN_DIR_PATH = self.data_dir
        cities_data_path_list = os.listdir(MAIN_DIR_PATH)

        #Beijing city data analysis
        sample_data_path = os.path.join(MAIN_DIR_PATH, cities_data_path_list[0])
        print(sample_data_path)
        data = self.load_data(sample_data_path)
        
        # Drop unnecessary columns first
        data = data.drop(columns=["No","PM_Dongsi", "PM_Dongsihuan", "PM_Nongzhanguan"])
                                    
        # Columns to drop at selection
        data = data.drop(columns= params_informer.drop_feat)
                                            
        # Rename "PM_US Post" to "PM"
        new_data = data.rename(columns={"PM_US Post": "PM"})
        
        # Handling 'cbwd' (Check if it exists first)
        if 'cbwd' in data.columns:
            
            # Define mapping (excluding 'cv' since it should be NaN)
            cbwd_mapping = {'NE': 1, 'NW': 2, 'SE': 3, 'SW': 4, 'cv' : 5}
    
            # Replace 'cv' with NaN and map other values
            new_data['cbwd'] = new_data['cbwd'].map(cbwd_mapping)
     
            # Convert to float to retain NaNs
            new_data['cbwd'] = new_data['cbwd'].astype(float)
             
        # Extract column names in order
        column_names = new_data.columns.tolist()
        
        # Columns in the dataset without date-time columns
        column_names_without_dt = column_names[4:]  # Slicing from index 4 onward
        
        # This step useful when we want to calculate loss for only the target cols out of all cols        
        target_columns = ["season", "PM", "DEWP", "HUMI", "PRES", "TEMP"]
        
        # Get indices of the target columns in column_names_without_dt
        target_column_indices = [column_names_without_dt.index(col) for col in target_columns if col in column_names_without_dt]
                      
        return new_data, column_names, target_column_indices
    
    def introduce_nans(self, data, nan_fraction=params_informer.nan_fraction):
        """
        Introduce NaNs randomly across all elements in a 3D dataset (batch, time, features).
    
        Parameters:
        - data: NumPy 3D array (batch, time, features) to modify.
        - nan_fraction: Fraction of elements to replace with NaNs.
    
        Returns:
        - NumPy array with NaNs introduced.
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a NumPy array.")
    
        if data.ndim != 3:
            raise ValueError(f"Expected a 3D array, got shape {data.shape}")
    
        new_data = data.copy()  # Work on a copy to avoid modifying the original dataset
        total_elements = new_data.size  # Total number of elements in the array
        num_nans = int(nan_fraction * total_elements)  # Number of NaNs to introduce
    
        if num_nans > 0:
            nan_indices = np.random.choice(total_elements, num_nans, replace=False)  # Pick random indices
            new_data.flat[nan_indices] = np.nan  # Assign NaNs at these positions
    
        return new_data

    
    def sort_time_series(self, new_data):
        
        # Sorting the data into 144 time series. Each time series contains only 1 year of data of an hour.
        # Each year has 24 separate time series (one for each hour of the day).

        # Define columns to keep (features + timestamps, including 'hour')
        keep_columns = ['year', 'month', 'day', 'hour'] + [col for col in new_data.columns if col not in ['year', 'month', 'day', 'hour']]

        # Sort data to ensure correct ordering
        new_data = new_data.sort_values(by=['year', 'month', 'day', 'hour']).reset_index(drop=True)

        # Remove February 29, 2012 data (leap day)
        new_data = new_data[~((new_data['year'] == 2012) & (new_data['month'] == 2) & (new_data['day'] == 29))]

        # Container to store each time series
        time_series_data = []

        # Extract unique years and iterate over each hour (0-23)
        years = sorted(new_data['year'].unique())  # Should be [2010, 2011, 2012, 2013, 2014, 2015]
        hours = list(range(24))  # Hours from 0 to 23

        # Loop through each hour and then through each year
        for hour in hours:
            for year in years:
                # Filter data for the specific hour and year
                df_filtered = new_data[(new_data['hour'] == hour) & (new_data['year'] == year)]
                
                # Ensure data is sorted correctly
                df_filtered = df_filtered.sort_values(by=['month', 'day'])

                # Select required columns (timestamps + features)
                data_array = df_filtered[keep_columns].values  # Shape should be (365, 14)

                # Append to list
                time_series_data.append(data_array)

        # Convert to PyTorch tensor
        time_series_tensor = torch.tensor(np.array(time_series_data), dtype=torch.float32)  # Shape: (144, 365, 14)

        # Verify final shape
        print(time_series_tensor.shape)  # Expected output: torch.Size([144, 365, 14])
       
        return time_series_data
    
    # Function to transform a dataset while keeping NaNs intact

    def scale_data (self, time_series_data):
        
        def get_year(ts):
            return int(ts[0, 0])  # First row, first column (year)
        
        # Split the data based on years
        train_indices = [i for i, ts in enumerate(time_series_data) if get_year(ts) in [2010, 2011, 2012, 2013]]
        val_indices = [i for i, ts in enumerate(time_series_data) if get_year(ts) == 2014]
        test_indices = [i for i, ts in enumerate(time_series_data) if get_year(ts) == 2015]
        
        train_data = [time_series_data[i] for i in train_indices]
        val_data = [time_series_data[i] for i in val_indices]
        test_data = [time_series_data[i] for i in test_indices]
    
        # Convert lists to numpy arrays
        train_array = np.array(train_data)  # Shape (N_train, 365, 14)
        val_array = np.array(val_data)      # Shape (N_val, 365, 14)
        test_array = np.array(test_data)    # Shape (N_test, 365, 14)
    
        # Identify feature columns (skip first 4: year, month, day, hour)
        feature_cols = slice(4, params_informer.data_feat)  # Selecting columns 4 to 14 (features only)
        
        # Initialize MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
    
        # Reshape for fitting: (N_train, 365, 10) → (N_train*365, 10)
        train_features = train_array[:, :, feature_cols].reshape(-1, data_ft)
        
        # Fit scaler on training data only
        scaler.fit(train_features)
    
        # Function to transform dataset while keeping NaNs intact
        def transform_data(data, scaler):
            data = np.array(data)  # Convert to numpy array if not already
            original_shape = data.shape  # Store original shape (N, 365, 14)
            
            # Flatten features for scaling: (N, 365, 10) → (N*365, 10)
            features = data[:, :, feature_cols].reshape(-1, data_ft)
            
            # Preserve NaNs during scaling
            nan_mask = np.isnan(features)
            features_scaled = scaler.transform(features)
            
            # Restore NaNs after transformation
            features_scaled[nan_mask] = np.nan
            
            # Reshape back to original shape (N, 365, 10)
            data[:, :, feature_cols] = features_scaled.reshape(original_shape[0], 365, data_ft)
            
            return data
    
        # Apply transformation to train, validation, and test sets
        train_scaled = transform_data(train_array, scaler)
        val_scaled = transform_data(val_array, scaler)
        test_scaled = transform_data(test_array, scaler)
    
        return train_scaled, val_scaled, test_scaled, scaler
    
    def time_features(self, dates, freq='h'):
        
        # Ensure dates is a copy to avoid SettingWithCopyWarning
        dates = dates.copy()

        # Extract time features using vectorized operation
        dates['month'] = dates.datetime.dt.month
        dates['day'] = dates.datetime.dt.day
        dates['weekday'] = dates.datetime.dt.weekday
        dates['hour'] = dates.datetime.dt.hour
        dates['minute'] = dates.datetime.dt.minute
        
        # Create 15-minute intervals for minute column (0, 1, 2, ..., 3 for each quarter of an hour)
        dates['minute'] = dates['minute'] // 15
    
        freq_map = {
            'y':[],'m':['month'],'w':['month'],
            'd':['month','day','weekday'],
            'b':['month','day','weekday'],
            'h':['month','day','weekday','hour'],
            't':['month','day','weekday','hour','minute'],
        }
        return dates[freq_map[freq.lower()]].values

    
    def filter_by_months(self, test_set, encoder_months, decoder_months, pred_months):
        """
        Filters test_set separately for encoder, decoder, and prediction based on selected months.
    
        Args:
            test_set (numpy array): Shape (samples, timesteps, features), where test_set[:, :, 1] represents the month.
            encoder_months (list): List of months for encoder input.
            decoder_months (list): List of months for decoder input.
            pred_months (list): List of months for target prediction.
    
        Returns:
            Tuple of numpy arrays: (encoder_inp, decoder_inp, target_pred), all with shape (samples, selected_timesteps, features)
        """
        batch_size, total_timesteps, num_features = test_set.shape  # Get shape info
    
        def apply_mask(data, months):
            """Applies month mask per sample while keeping 3D shape."""
            return np.array([sample[np.isin(sample[:, 1], months)] for sample in data], dtype=object)
    
        # Apply masks while keeping batch dimension
        encoder_inp = apply_mask(test_set, encoder_months)
        decoder_inp = apply_mask(test_set, decoder_months)
        target_pred = apply_mask(test_set, pred_months)
    
        return encoder_inp, decoder_inp, target_pred


    def prepare_informer_data(self, input_sequences, seq_len, label_len, pred_len):
        """
        Prepares data for the Informer model by iterating through the input sequences and 
        splitting them into encoder and decoder sequences.

        Args:
            input_sequences (numpy array): Shape (N, total_timesteps, 14), where N is the number of samples.
            seq_len (int): Encoder sequence length (365)
            label_len (int): Decoder known label length (185)
            pred_len (int): Prediction length (180)

        Returns:
            encoder_x, decoder_x, output_y, encoder_mark, decoder_mark (numpy arrays)
        """
        encoder_inp, decoder_inp, output_gt = [], [], []
        encoder_stamp, decoder_stamp = [], [] 
        # Store original timestamps
        dec_org_dtstmp, op_org_dtstmp = [], []
        
        # Reshape input into 2D for easier processing (N * total_timesteps, 14)
        batch_size, total_timesteps, num_features = input_sequences.shape
        input_sequences_2d = input_sequences.reshape(-1, num_features)  # (N * total_timesteps, 14)

        # Extract year, month, day, hour (first 4 columns) and convert to datetime
        datetime_values = input_sequences_2d[:, :4].astype(int)
        date_df = pd.DataFrame(datetime_values, columns=['year', 'month', 'day', 'hour'])
        date_df["datetime"] = pd.to_datetime(date_df)

        # Compute time features
        data_stamp = self.time_features(date_df[['datetime']])  # Shape: (N * total_timesteps, time_features_dim)

        # Define feature slice (skip first 4 columns: year, month, day, hour)
        feature_cols = slice(4, params_informer.data_feat)

        # Iterate through input_sequences_2d to create overlapping sequences
        for i in range(0, len(input_sequences_2d) - seq_len - pred_len, seq_len+label_len+pred_len):    
            # Define start and end indices
            enc_start, enc_end = i, i + seq_len
            dec_start, dec_end = enc_end, enc_end + label_len + pred_len
            out_start, out_end = enc_end + label_len, enc_end + label_len + pred_len

            # Ensure we have enough data to construct both input and output
            if out_end > len(input_sequences_2d):
                break

            # Extract encoder input
            encoder_x = input_sequences_2d[enc_start:enc_end, feature_cols]

            # Prepare decoder input: known label_len values + placeholders (zeros for pred_len)
            decoder_x = np.zeros((label_len + pred_len, data_ft))
            decoder_x[:label_len, :] = input_sequences_2d[dec_start:dec_start + label_len, feature_cols]

            # Extract target output
            output_y = input_sequences_2d[out_start:out_end, feature_cols]

            # Extract timestamps for encoder and decoder
            encoder_mark = data_stamp[enc_start:enc_end, :]
            decoder_mark = data_stamp[dec_start:dec_end, :]
            
            # Extract original datetime values and convert to string
            dec_org_tstamp = date_df["datetime"].iloc[dec_start:dec_end].astype(str).values.reshape(-1, 1)
            op_org_tstamp = date_df["datetime"].iloc[out_start:out_end].astype(str).values.reshape(-1, 1)

            # **Ensure consistent dimensions before appending**
            if (
                encoder_x.shape[0] == seq_len and 
                decoder_x.shape[0] == label_len + pred_len and 
                output_y.shape[0] == pred_len and 
                encoder_mark.shape[0] == seq_len and 
                decoder_mark.shape[0] == label_len + pred_len
            ):
                encoder_inp.append(encoder_x)
                decoder_inp.append(decoder_x)
                output_gt.append(output_y)
                encoder_stamp.append(encoder_mark)
                decoder_stamp.append(decoder_mark)
                
                dec_org_dtstmp.append(dec_org_tstamp)
                op_org_dtstmp.append(op_org_tstamp)

        # Convert to numpy arrays and ensure consistent shape
        encoder_inp = np.array(encoder_inp, dtype=object)
        decoder_inp = np.array(decoder_inp, dtype=object)
        output_gt = np.array(output_gt, dtype=object)
        encoder_stamp = np.array(encoder_stamp, dtype=object)
        decoder_stamp = np.array(decoder_stamp, dtype=object)
    
        dec_org_dtstmp = np.array(dec_org_dtstmp, dtype=object)
        op_org_dtstmp = np.array(op_org_dtstmp, dtype=object)

        return encoder_inp, decoder_inp, output_gt, encoder_stamp, decoder_stamp, dec_org_dtstmp, op_org_dtstmp
    
    
    def prepare_data(self):
        # Load and clean data
        cities_data_path_list = os.listdir(self.data_dir)
        sample_data_path = os.path.join(self.data_dir, cities_data_path_list[0])
        raw_data = self.load_data(sample_data_path)
        processed_data, column_names, target_column_indices = self.clean_and_process_data(raw_data)
        
        # Sorting the data into various time series sets
        time_series_data = self.sort_time_series(processed_data)
        
        # Split and standardize
        train_set, val_set, test_set, scaler = self.scale_data(time_series_data)
                             
        # Define custom months to send into encoder, decoder and testing for train, val and test sets
        encoder_months = params_informer.encoder_months
        decoder_months = params_informer.decoder_months
        pred_months = params_informer.pred_months
        
       
        # Custom months selection    
        filtered_enc_train, filtered_dec_train, filtered_pred_train = self.filter_by_months(train_set, encoder_months, decoder_months, pred_months)
        filtered_enc_val, filtered_dec_val, filtered_pred_val = self.filter_by_months(val_set, encoder_months, decoder_months, pred_months)
        filtered_enc_test, filtered_dec_test, filtered_pred_test = self.filter_by_months(test_set, encoder_months, decoder_months, pred_months)
        
        # Ensure correct sizes before stacking
        if filtered_enc_test.shape[-2] + filtered_dec_test.shape[-2] + filtered_pred_test.shape[-2] == seq_len+label_len+pred_len:
            filtered_train_set = np.concatenate([filtered_enc_train, filtered_dec_train, filtered_pred_train], axis=1)
            filtered_val_set = np.concatenate([filtered_enc_val, filtered_dec_val, filtered_pred_val], axis=1)
            filtered_test_set = np.concatenate([filtered_enc_test, filtered_dec_test, filtered_pred_test], axis=1)
            
        else:
            raise ValueError("Filtered data does not match expected sequence lengths!")
    
           
        # Prepare test data using the merged train_set & test_set
        
        X_train_encc, X_train_decoderr, y_train, X_train_enc_mark, X_train_dec_mark, dec_train_org_dtst, op_train_org_dtst = self.prepare_informer_data(
                                                                                        filtered_train_set, seq_len, label_len, pred_len)
        
        X_val_encc, X_val_decoderr, y_val, X_val_enc_mark, X_val_dec_mark, dec_val_org_dtst, op_val_org_dtst = self.prepare_informer_data(
                                                                                        filtered_val_set, seq_len, label_len, pred_len)
        
        X_test_encc, X_test_decoderr, y_test, X_test_enc_mark, X_test_dec_mark, dec_test_org_dtst, op_test_org_dtst = self.prepare_informer_data(
                                                                                        filtered_test_set, seq_len, label_len, pred_len)
    
                
        # Only introduce NaNs to input data (not into ground-truth and datestamps)
        X_train_enc = self.introduce_nans(X_train_encc)
        X_train_decoder = self.introduce_nans(X_train_decoderr)    
        X_val_enc = self.introduce_nans(X_val_encc)
        X_val_decoder = self.introduce_nans(X_val_decoderr)
        X_test_enc = self.introduce_nans(X_test_encc)
        X_test_decoder = self.introduce_nans(X_test_decoderr)
        
        
        # Convert to tensors       
        train_enc_tensor = torch.tensor(np.array(X_train_enc, dtype=np.float32), dtype=torch.float32)
        train_dec_tensor = torch.tensor(np.array(X_train_decoder, dtype=np.float32), dtype=torch.float32)
        train_enc_mark_tns = torch.tensor(np.array(X_train_enc_mark, dtype=np.float32), dtype=torch.float32)
        train_dec_mark_tns = torch.tensor(np.array(X_train_dec_mark, dtype=np.float32), dtype=torch.float32)
        train_output_gt = torch.tensor(np.array(y_train, dtype=np.float32), dtype=torch.float32)
      
        val_enc_tensor = torch.tensor(np.array(X_val_enc, dtype=np.float32), dtype=torch.float32)
        val_dec_tensor = torch.tensor(np.array(X_val_decoder, dtype=np.float32), dtype=torch.float32)
        val_enc_mark_tns = torch.tensor(np.array(X_val_enc_mark, dtype=np.float32), dtype=torch.float32)
        val_dec_mark_tns = torch.tensor(np.array(X_val_dec_mark, dtype=np.float32), dtype=torch.float32)
        val_output_gt = torch.tensor(np.array(y_val, dtype=np.float32), dtype=torch.float32)
                
        test_enc_tensor = torch.tensor(np.array(X_test_enc, dtype=np.float32), dtype=torch.float32)
        test_dec_tensor = torch.tensor(np.array(X_test_decoder, dtype=np.float32), dtype=torch.float32)
        test_enc_mark_tns = torch.tensor(np.array(X_test_enc_mark, dtype=np.float32), dtype=torch.float32)
        test_dec_mark_tns = torch.tensor(np.array(X_test_dec_mark, dtype=np.float32), dtype=torch.float32)
        test_output_gt = torch.tensor(np.array(y_test, dtype=np.float32), dtype=torch.float32)
        test_dec_org_dtstmp = np.array(dec_test_org_dtst)
        test_op_org_dtstmp = np.array(op_test_org_dtst)
        
        scaler = scaler      
        column_names = column_names
        target_column_indices = target_column_indices

        return (train_enc_tensor, train_dec_tensor, train_enc_mark_tns, train_dec_mark_tns, train_output_gt,
                val_enc_tensor, val_dec_tensor, val_enc_mark_tns, val_dec_mark_tns, val_output_gt,
                test_enc_tensor, test_dec_tensor, test_enc_mark_tns, test_dec_mark_tns, test_output_gt, 
                test_dec_org_dtstmp, test_op_org_dtstmp, scaler, column_names, target_column_indices)
    
               
# # To try the DataPreparer class
# # Create an instance of DataPreparer
# data_preparer = DataPreparer(data_dir='../01_PM2.5 Chinese Weather data')

# data_preparer.prepare_data()
