import numpy as np
import pandas as pd
import src.pipelines.data_load as ld
import src.pipelines.preprocessing as pps
from pycaret.regression import load_model, predict_model
from pycaret.regression import *
import configparser
# import lime
# import lime.lime_tabular

config_path='./config/config'
config = configparser.ConfigParser()
config.read(config_path)

def get_score(test_df):
    """calculate the readability score using the machine learning model

    Args:
        test_df (dataframe): the input text in a dataframe format passed on by display.py 

    Returns:
        float: The readability score
    """
    # Pass the text through the preprocessing pipeline
    pps_test_df = pps.preprocess(test_df, "excerpt")
    model_folder = config.get('TRAIN','MODEL')
    # Load model
    #model = load_model('/home/kaushlesh/CommonLit Readability/model/best_model')
    model = load_model(model_folder+"/"+"best_model")
    # Predict the redability score
    score=predict_model(model, data=pps_test_df)
       
    return score["Label"][0]