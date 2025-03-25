In this study, we use LSTM and Informer networks to predict all input features of the Beijing city dataset.

Project codebase structure overview 
 
Brief documentation of the project codebase structure.

01_PM2.5 Chinese Weather data – The original Chinese weather datasets of various cities (Excel format) 
BeijingPM20100101_20151231 – This dataset is used in this project.
ChengduPM20100101_20151231
GuangzhouPM20100101_20151231
ShanghaiPM20100101_20151231
ShenyangPM20100101_20151231

Data 
Data_anlysis_imgs – The images showing all the data of each feature from the dataset (2010 - 2015)
Prepare_data_inf – This file is responsible for the preparation of the data for the Informer model
Prepare_data_lstm – This file is responsible for the preparation of the data for the LSTM model

Drafts – This folder contains the files for running the informer model in sequential form (former model) when data is not divided into 144 time-series based on hourly data of each year.

Miscll
Study_folder – This file is responsible for creating a local study folder for each run of the model to store the results and other data.

Network
Inf_network – The folder containing the python scripts responsible for creating an Informer model
Utils - (short for "utilities") folder typically contains helper functions, scripts, and tools that support the main Informer model's functionality.
Lstm_network – The python script of the LSTM network

Params_informer – The hyperparameters file responsible for running the Informer model
Params_lstm – The hyperparameters file responsible for running the LSTM model
Train_test_informer – The main file responsible for training and testing the Informer model. (This file should be run to start the Informer model training/testing)
Train_test_lstm – The main file responsible for training and testing the LSTM model. (This file should be run to start the LSTM model training/testing)
