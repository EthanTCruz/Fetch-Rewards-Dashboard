from pydantic_settings import BaseSettings

class Settings(BaseSettings, case_sensitive=True):
    dataFileLocation: str = 'data\data_daily.csv'