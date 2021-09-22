import numpy as np
import pandas as pd
import configparser
import src.pipelines.data_load as ld
import src.pipelines.preprocessing as pps
import os
import sys
from pycaret.regression import *

# The path to configuration file
config_path='./config/config'

train_df = ld.load_data(config_path)

config = configparser.ConfigParser()
config.read(config_path)

# Drop the columns not needed for training
drop_columns = config.get('COLUMNS','COLUMNS_TO_DROP').split(',')
if not drop_columns:
    print("No columns specified to be dropped")
    prep_train_df = train_df
else:
    prep_train_df = train_df.drop(drop_columns,axis='columns')

# Identify the column with the text
text_column = config.get('COLUMNS','TEXT_COLUMN')
if not text_column:
    sys.exit("No column with text specified")
else:
    excerpt = text_column

# Identify the column with the readability scores
target_column = config.get('COLUMNS','TARGET_COLUMN')
if not target_column:
    sys.exit("No target column specified")
else:
    target = target_column

# Preprocess the data to create features
pps_train_df = pps.preprocess(prep_train_df, excerpt)
#Storing the data in an intermediate file for model experimentation as preprocessing takes a large amount of time
pps_train_df.to_csv("intermediate_df.csv")
# Drop the column containig the text
final_df=pps_train_df.drop(['excerpt'],axis='columns')


# pps_train_df = pd.read_csv("intermediate_df.csv")
# print(pps_train_df)
# final_df=pps_train_df.drop(['Unnamed: 0','excerpt'],axis='columns')

# Seting up the pycaret environment for AutoML
exp_reg102 = setup(data = final_df, target = 'target', session_id=123,
                  normalize = True, transformation = True, transform_target = False,
                  combine_rare_levels = False, rare_level_threshold = 0.05,
                  remove_multicollinearity = True, multicollinearity_threshold = 0.95,
                  create_clusters= True, polynomial_features= True, remove_outliers=True,
                  feature_interaction= True, silent=True, transform_target_method= 'yeo-johnson',
                  log_experiment = False, experiment_name = 'readability')

# compare all baseline models and select top 5
top5 = compare_models(n_select = 5)

# tune top 5 base models
tuned_top5 = [tune_model(i) for i in top5]

# ensemble top 5 tuned models
bagged_top5 = [ensemble_model(i) for i in tuned_top5]

# blend top 5 base models 
blender = blend_models(estimator_list = top5) 

# select best model 
best = automl(optimize = 'R2')

# If a Summary plot can be generated
for i in top5:
    try:
        interpret_model(i)

    except Exception as e:
        print(e)
        continue

model_folder = config.get('TRAIN','MODEL')
# save best model
save_model(best, model_folder+"/"+"best_model")



