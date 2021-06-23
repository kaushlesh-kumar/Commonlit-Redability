from kaggle.api.kaggle_api_extended import KaggleApi
import configparser
import os

config = configparser.ConfigParser()
config.read('./config/config')

api = KaggleApi()
api.authenticate()
print(config.get('KAGGLE','COMPETITION'))
api.competition_download_files(config.get('KAGGLE','COMPETITION'), path = config.get('KAGGLE','DATA_FOLDER'))
filepath = config.get('KAGGLE','DATA_FOLDER')+config.get('KAGGLE','DOWNLOAD_FILE')
from zipfile import ZipFile
zf = ZipFile(filepath)
zf.extractall(config.get('KAGGLE','DATA_FOLDER')) #save files in selected folder
zf.close()

os.remove(filepath) 