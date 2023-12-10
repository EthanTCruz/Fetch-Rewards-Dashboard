import pandas as pd
import plotly.graph_objects as go
from arima_model import ArimaPredictor
from prophet_model import ProphetPredictor
from linear_regression_model import LinearRegressionPredictor
from Recurrent_Neureal_Network import RNNPredictor
from LSTM_Recurrent_Neureal_Network import LSTMPredictor
from pathlib import Path
import json

from config import Settings

s = Settings()

class Grapher():
    def __init__(self,dataFileLocation: str = s.dataFileLocation) -> None:
        self.dataFileLocation = dataFileLocation

    def returnMonthlyData(self):
        data_csv = pd.read_csv(self.dataFileLocation)

        data_csv['Date'] = pd.to_datetime(data_csv['# Date'])
        data_csv.drop(columns=['# Date'], inplace=True)

        data_monthly = data_csv.groupby(pd.Grouper(key='Date', freq='M')).sum()
        return data_monthly

    def Graph(self,predicted_data, data_monthly, predicted_months: pd.DatetimeIndex):
        CI_stats = predicted_data
        predicted_data = predicted_data["monthly_sums"]
        summary_statistics = {}
        summary_statistics["Projected 2022 Monthly Mean"] = round(predicted_data["Predicted_Receipts"].mean(),2)
        summary_statistics["Projected 2022 Monthly Min"] = int(round(predicted_data["Predicted_Receipts"].min(),0))
        summary_statistics["Projected 2022 Monthly Max"] = int(round(predicted_data["Predicted_Receipts"].max(),0))
        summary_statistics["Projected 2022  Sum"] = int(round(predicted_data["Predicted_Receipts"].sum(),0))

        summary_statistics["2021 Monthly Mean"] = round(data_monthly["Receipt_Count"].mean(),2)
        summary_statistics["2021 Monthly Min"] = int(round(data_monthly["Receipt_Count"].min(),0))
        summary_statistics["2021 Monthly Max"] = int(round(data_monthly["Receipt_Count"].max(),0))
        summary_statistics["2021 Sum"] = int(round(data_monthly["Receipt_Count"].sum(),0))

        # Create Plotly figure
        fig = go.Figure()

        # Add predicted data
        fig.add_trace(go.Scatter(x=predicted_months, y=predicted_data['Predicted_Receipts'].values, 
                                mode='lines+markers', name='Projected 2022 Data', 
                                line=dict(color='red', dash='dash')))
        conf_key = "conf_int"
        if CI_stats.get(conf_key) is not None:
            lower_ci = CI_stats["conf_int"]["lower Receipt_Count"]
            upper_ci = CI_stats["conf_int"]["upper Receipt_Count"]

                    # Add lower bound of confidence interval
            fig.add_trace(go.Scatter(x=predicted_months, y=lower_ci, 
                                    mode='lines', name='Lower Bound',
                                    line=dict(width=0),
                                    showlegend=False))

            # Add upper bound of confidence interval and fill the area
            fig.add_trace(go.Scatter(x=predicted_months, y=upper_ci, 
                                    mode='lines', name='Upper Bound',
                                    line=dict(width=0),
                                    fill='tonexty',  # Fill area between the confidence interval bounds
                                    fillcolor='rgba(255, 0, 0, 0.3)',  # Light red fill with some transparency
                                    showlegend=False))


        # Add actual data
        fig.add_trace(go.Scatter(x=data_monthly.index, y=data_monthly['Receipt_Count'].values, 
                                mode='lines+markers', name='2021 Data',
                                line=dict(color='blue', dash='solid')))

        
        # Update layout
        fig.update_layout(title='Projected 2022 Monthly Scans',
                        xaxis_title='Month',
                        yaxis_title='Receipts Scanned (per month)',
                        template='plotly_dark')
        fig.update_layout(
            autosize=True
        )


        return fig, summary_statistics

    def GenerateGraphs(self,predicted_data):
        temp = predicted_data
        predicted_data = predicted_data["monthly_sum"]
        data_monthly = self.returnMonthlyData()
        predicted_months = pd.date_range(start=data_monthly.index[-1], periods=13, freq='M')

        new_data = pd.DataFrame({
                        'Predicted_Receipts': data_monthly.values[-1],
                        'Date': [pd.to_datetime(data_monthly.index[-1])]
                                })
        
        predicted_data = pd.concat([new_data, predicted_data])
        conf_key = "conf_int"
        if temp.get(conf_key) is not None:
            new_data = pd.DataFrame({
                    'upper Receipt_Count': data_monthly.values[-1],
                    'lower Receipt_Count': data_monthly.values[-1],
                    'Date': [pd.to_datetime(data_monthly.index[-1])]
                            })
            
            new_data['Date'] = pd.to_datetime(new_data['Date'])
            new_data.set_index('Date', inplace=True)

            temp["conf_int"] = pd.concat([new_data, temp["conf_int"]], axis=0)


        temp["monthly_sums"] = predicted_data
        graph = self.Graph(predicted_data=temp,predicted_months=predicted_months,data_monthly=data_monthly)

        return graph


    def LinearRegressionGraphs(self):
        lrm = LinearRegressionPredictor(dataFileLocation=self.dataFileLocation)
        predicted_data = lrm.predict_by_months()
        return self.GenerateGraphs(predicted_data=predicted_data)

    def ArimaGraphs(self):
        arima = ArimaPredictor(dataFileLocation=self.dataFileLocation)
        predicted_data = arima.PredictNMonths()
        return self.GenerateGraphs(predicted_data=predicted_data)

    def ProphetGraphs(self):
        prophet = ProphetPredictor(dataFileLocation=self.dataFileLocation)
        predicted_data = prophet.PredictNDays()
        return self.GenerateGraphs(predicted_data=predicted_data)

    def RNNGraphs(self,refresh_data: bool = False):
        path = Path(s.simple_rnn_prediction)
        if path.exists() and not refresh_data:
            with open(s.simple_rnn_prediction, 'r') as json_file:
                predicted_data = json.load(json_file)
                json_str = predicted_data["monthly_sum"]
                df = pd.DataFrame(json.loads(json_str))

                # Converting the 'Date' column from Unix timestamp (in milliseconds) to datetime
                df['Date'] = pd.to_datetime(df['Date'], unit='ms')
                predicted_data["monthly_sum"] = df
                conf_key = "conf_int"
                if predicted_data.get(conf_key) is not None:
                    predicted_data[conf_key] = df[conf_key].to_json()

        else:
            with open(s.simple_rnn_prediction, 'w') as json_file:
                rnn = RNNPredictor()
                predicted_data = rnn.predict()
                temp = predicted_data.copy()
                temp["monthly_sum"] = temp["monthly_sum"].to_json()
                conf_key = "conf_int"
                if temp.get(conf_key) is not None:
                    temp[conf_key] = temp[conf_key].to_json()

                json.dump(temp, json_file)

        return self.GenerateGraphs(predicted_data=predicted_data)
    

    def LSTMGraphs(self,refresh_data: bool = False):
        path = Path(s.simple_rnn_prediction)
        if path.exists() and not refresh_data:
            with open(s.lstm_rnn_prediction, 'r') as json_file:
                predicted_data = json.load(json_file)
                json_str = predicted_data["monthly_sum"]
                df = pd.DataFrame(json.loads(json_str))

                # Converting the 'Date' column from Unix timestamp (in milliseconds) to datetime
                df['Date'] = pd.to_datetime(df['Date'], unit='ms')
                predicted_data["monthly_sum"] = df
                conf_key = "conf_int"
                if predicted_data.get(conf_key) is not None:
                    predicted_data[conf_key] = df[conf_key].to_json()

        else:
            with open(s.lstm_rnn_prediction, 'w') as json_file:
                lstm = LSTMPredictor()
                predicted_data = lstm.predict()
                temp = predicted_data.copy()
                temp["monthly_sum"] = temp["monthly_sum"].to_json()
                conf_key = "conf_int"
                if temp.get(conf_key) is not None:
                    temp[conf_key] = temp[conf_key].to_json()

                json.dump(temp, json_file)

        return self.GenerateGraphs(predicted_data=predicted_data)
    