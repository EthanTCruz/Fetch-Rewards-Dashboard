from config import Settings
from arima_model import ArimaPredictor
from GenerateGraphs import Grapher

s = Settings()
dataFileLocation = s.dataFileLocation
grapher=Grapher(dataFileLocation=dataFileLocation)


def main():
    # a = ArimaPredictor(dataFileLocation=dataFileLocation)
    # a.PredictNMonths()
    grapher.ProphetGraphs()
    return 0









if __name__ == "__main__":
    main()