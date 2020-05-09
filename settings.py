import os
import pandas as pd

COMPETITION = 'm5-forecasting-accuracy'

DIR_DATA = 'input_data'
NOTEBOOK_DIR = 'notebooks'
OUTPUT_DIR = 'output'

#output data 
FEATURE_DIR = os.path.join(OUTPUT_DIR, 'feature')
METRIC_DIR = os.path.join(OUTPUT_DIR, 'metric')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'model')

# output data - directories for the cross validation and ensembling
VAL_DIR = os.path.join(OUTPUT_DIR, 'val')
TST_DIR = os.path.join(OUTPUT_DIR, 'tst')
SUBMISSION_DIR = os.path.join(OUTPUT_DIR, 'submission')

#input data
MAIN_DATA = os.path.join(DIR_DATA, 'sales_train_validation.csv')
TRAIN_DATA = os.path.join(DIR_DATA, 'train.feather')
TEST_DATA = os.path.join(DIR_DATA, 'test.feather')
SAMPLE_SUBMISSION = os.path.join(DIR_DATA, 'sample_submission.csv')
PRICES_DATA = os.path.join(DIR_DATA, 'sell_prices.csv')
CALENDAR_DATA = os.path.join(DIR_DATA, 'calendar.csv')

#constants
TARGET = 'sales'            # Our target
STORES_IDS = pd.read_csv(MAIN_DATA)['store_id']
STORES_IDS = list(STORES_IDS.unique())

# #LIMITS and const

# START_TRAIN = 1069               # We can skip some rows (Nans/faster training)
# END_TRAIN   = 1913               # End day of our train set
# P_HORIZON   = 28                 # Prediction horizon
# USE_AUX     = True               # Use or not pretrained models

# #FEATURES to remove
# ## These features lead to overfit
# ## or values not present in test set
# remove_features = ['id','state_id','store_id',
#                    'date','wm_yr_wk','d',TARGET]
# mean_features   = ['enc_cat_id_mean','enc_cat_id_std',
#                    'enc_dept_id_mean','enc_dept_id_std',
#                    'enc_item_id_mean','enc_item_id_std'] 


# # AUX(pretrained) Models paths
# AUX_MODELS = '../input/m5-aux-models/'


# #STORES ids
# STORES_IDS = pd.read_csv(ORIGINAL+'sales_train_validation.csv')['store_id']
# STORES_IDS = list(STORES_IDS.unique())


# #SPLITS for lags creation
# SHIFT_DAY  = 28
# N_LAGS     = 15
# LAGS_SPLIT = [col for col in range(SHIFT_DAY,SHIFT_DAY+N_LAGS)]
# ROLS_SPLIT = []
# for i in [1,7,14]:
#     for j in [7,14,30,60]:
#         ROLS_SPLIT.append([i,j])