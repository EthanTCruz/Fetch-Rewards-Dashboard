from arima_model import ArimaPredictor
from prophet_model import ProphetPredictor
from config import Settings
import pandas as pd
import plotly.graph_objects as go

from linear_regression_model import LinearProgressionPredictor
from GenerateGraphs import Grapher

s = Settings()
dataFileLocation = s.dataFileLocation
grapher=Grapher(dataFileLocation=dataFileLocation)


def main():
    return 0



def LinearRegressionGraphs():
    lrm = LinearProgressionPredictor(dataFileLocation='data\data_daily.csv')
    predicted_data = lrm.predict_by_months()
    graph_name='lrm.png'
    return grapher.GenerateGraphs(predicted_data=predicted_data,graph_name=graph_name)


def ArimaGraphs():
    arima = ArimaPredictor(dataFileLocation='data\data_daily.csv')
    predicted_data = arima.PredictNMonths()
    graph_name = 'arima.png'
    return grapher.GenerateGraphs(predicted_data=predicted_data,graph_name=graph_name)

def ProphetGraphs():
    prophet = ProphetPredictor(dataFileLocation='data\data_daily.csv')
    predicted_data = prophet.PredictNDays()
    graph_name = 'prophet.png'
    return grapher.GenerateGraphs(predicted_data=predicted_data,graph_name=graph_name)






if __name__ == "__main__":
    main()