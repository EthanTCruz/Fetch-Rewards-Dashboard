from config import Settings
from arima_model import ArimaPredictor
from GenerateGraphs import Grapher
from Recurrent_Neureal_Network import RNNPredictor
from LSTM_Recurrent_Neureal_Network import LSTMPredictor

s = Settings()
dataFileLocation = s.dataFileLocation
grapher=Grapher(dataFileLocation=dataFileLocation)


def main():
    #a = ArimaPredictor(dataFileLocation=dataFileLocation)
    
    #grapher.ProphetGraphs()
    #grapher.ArimaGraphs()
    # r = RNNPredictor(step_size=7)
    # out = r.predict()
    #grapher.RNNGraphs(refresh_data=False)
    #put = a.PredictNMonths()
    l = LSTMPredictor()
    l.predict(train_model=False)

    return 0









if __name__ == "__main__":
    main()