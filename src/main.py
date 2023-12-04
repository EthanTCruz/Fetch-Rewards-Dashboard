from arima_model import ArimaPredictor
from prophet_model import ProphetPredictor
from linear_regression_model import LinearProgressionPredictor
from config import Settings

from GenerateGraphs import Grapher

s = Settings()
dataFileLocation = s.dataFileLocation
grapher=Grapher(dataFileLocation=dataFileLocation)


def main():
    return 0









if __name__ == "__main__":
    main()