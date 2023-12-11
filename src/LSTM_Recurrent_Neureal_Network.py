import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd
import numpy as np
from config import Settings
import math
from tqdm import tqdm
import random
import logging
tf.get_logger().setLevel(logging.ERROR)

s = Settings()

class LSTMPredictor:
    def __init__(self,dataFileLocation: str = s.dataFileLocation,step_size: int = 7) -> None:
        self.dataFileLocation = dataFileLocation
        self.step_size = step_size

    def column_month_summation(self,df,column_name,start_date: str ='2022-01-01'):
        temp_df = pd.DataFrame({'Date': pd.date_range(start=start_date, periods=365, freq='D'),
                            'Predicted_Receipts': df[column_name]})
        df['Month'] = temp_df['Date'].dt.to_period('M').dt.to_timestamp('M')
        monthly_sum = df.groupby('Month')[column_name].sum().reset_index()
        monthly_sum['Date'] = monthly_sum['Month']
        monthly_sum.drop(columns=['Month'], inplace=True)
        df = monthly_sum
        return df

    def create_sequences(self, data, n_steps):
        X, y = [], []
        for i in range(len(data)):
            end_ix = i + n_steps
            if end_ix >= len(data):
                break
            seq_x, seq_y = data[i:end_ix], data[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def split_data(self,data):
        X, y = self.create_sequences(data['Normalized_Receipt_Count'].values, self.step_size)

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]


        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


        return X_train, X_test, y_train, y_test


    def Train_Model(self):
        random.seed(3141)
        d_min,d_max,data_csv = self.clean_data()
        X_train, X_test, y_train, y_test = self.split_data(data=data_csv)

        time_steps = self.step_size
        n_features = 1

        model = Sequential()
        #tanh was the only activation function not overfitting, could be a bigger issue, but it's beyond my expertise at the moment
        model.add(LSTM(50, activation='tanh', return_sequences=True, input_shape=(time_steps, n_features)))
        model.add(LSTM(50, activation='tanh'))

        model.add(Dense(self.step_size))
        model.compile(optimizer='adam', loss='mse')

        model.fit(X_train, y_train, epochs=100, batch_size=32,verbose=0)
        #denormalize/normalize for evaluation?
        #loss = model.evaluate(X_test, y_test)
        #predictions = model.predict(X_test)

        tf.keras.models.save_model(model=model,filepath=s.LSTMModelFile)
        return model

    def clean_data(self):
        data_csv = pd.read_csv(self.dataFileLocation)
        data_csv['Date'] = pd.to_datetime(data_csv['# Date'])
        data_csv.drop(columns=['# Date'], inplace=True)

        d_max = data_csv['Receipt_Count'].max()
        d_min = data_csv['Receipt_Count'].min()
        data_csv['Normalized_Receipt_Count'] = (data_csv['Receipt_Count'] - d_min) / (d_max - d_min)

        return d_min,d_max,data_csv


    def predict(self,train_model:bool = True):
        if train_model:
            model = self.Train_Model()
        else:
            model = tf.keras.models.load_model(filepath=s.LSTMModelFile)

        d_min,d_max,data_csv = self.clean_data()


        last_n_days_data = data_csv['Normalized_Receipt_Count'].values[-self.step_size:]
        predicted_receipt_counts = []
        current_sequence = last_n_days_data.reshape((1, self.step_size, 1))
        number_of_prediction_steps = math.ceil(365/self.step_size)
        
        for i in tqdm(range(number_of_prediction_steps)):
            next_days_prediction = model.predict(current_sequence,verbose=0)
            next_days_prediction = next_days_prediction.reshape(1,self.step_size,1)
            current_sequence = next_days_prediction

            predicted_receipt_counts += list(next_days_prediction.flatten())

        results = [(x * (d_max - d_min)) + d_min for x in predicted_receipt_counts]
        data_csv['Predicted_Receipts'] = results[0:365]
        monthly_sum = self.column_month_summation(df = data_csv,column_name='Predicted_Receipts')
        results = {"monthly_sum":monthly_sum}
        
        return (results)