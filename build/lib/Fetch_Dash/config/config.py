from pydantic_settings import BaseSettings

class Settings(BaseSettings, case_sensitive=True):
    data_dir: str = './data'
    dataFileLocation: str = f'{data_dir}/raw/data_daily.csv'
    HOST: str = "127.0.0.1"
    arima_order: tuple = (-1,-1,-1,-1)
    arima_seasonal_order: tuple = (-1,-1,-1,-1)
    arima_default: tuple = (-1,-1,-1,-1)
    models_dir: str = f'{data_dir}/models'
    RNNModelFile: str = f'{models_dir}/rnn_model'
    LSTMModelFile: str = f'{models_dir}/lstm_model'
    predicted_dir: str = f'{data_dir}/processed'
    simple_rnn_prediction: str = f'{predicted_dir}/simple_rnn_predictions'
    arima_prediction: str = f'{predicted_dir}/arima_predictions'
    linear_regression_prediction: str = f'{predicted_dir}/lr_predictions'
    prophet_prediction: str = f'{predicted_dir}/prophet_predictions'
    lstm_rnn_prediction: str = f'{predicted_dir}/lstm_predictions'
    cache_dir: str = f'{data_dir}/cache-directory'
