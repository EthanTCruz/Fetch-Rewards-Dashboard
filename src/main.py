from config import Settings
from arima_model import ArimaPredictor
from GenerateGraphs import Grapher
from Recurrent_Neureal_Network import RNNPredictor

s = Settings()
dataFileLocation = s.dataFileLocation
grapher=Grapher(dataFileLocation=dataFileLocation)


def main():
    #a = ArimaPredictor(dataFileLocation=dataFileLocation)
    
    #grapher.ProphetGraphs()
    #grapher.ArimaGraphs()
    r = RNNPredictor(step_size=7)
    out = r.Predict()
    #put = a.PredictNMonths()

    return 0









if __name__ == "__main__":
    main()