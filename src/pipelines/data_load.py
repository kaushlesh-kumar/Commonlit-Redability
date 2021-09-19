import pandas as pd

import configparser

def load_data(config_path):

    config = configparser.ConfigParser()
    config.read(config_path)

    filepath = config.get('TRAIN','DATA_FOLDER')+config.get('TRAIN','TRAIN_DATA')
    train_df = pd.read_csv(filepath)

    return train_df