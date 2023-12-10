from pydantic_settings import BaseSettings

class Settings(BaseSettings, case_sensitive=True):
    dataFileLocation: str = '.\data\data_daily.csv'
    HOST: str = "127.0.0.1"
    arima_order: tuple = (-1,-1,-1,-1)
    arima_seasonal_order: tuple = (-1,-1,-1,-1)
    arima_default: tuple = (-1,-1,-1,-1)
    ModelFile: str = './rnn_model'