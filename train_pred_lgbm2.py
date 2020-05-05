import lightgbm as lgb
import pandas as pd
import datetime
import pickle
import random
import numpy as np
import time
import os
import gc
import settings
from sklearn.metrics import mean_squared_error
from math import sqrt
from utils import save_score

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed) 

def train(feature_name, model_name, lgb_params):

    #load initial train set
    train_Poe = pd.read_feather(os.path.join(settings.FEATURE_DIR, '{0}.trn.feather'.format('Poe')))
    train_simple = pd.read_feather(os.path.join(settings.FEATURE_DIR, '{0}.trn.feather'.format('simple')))
    train_df = pd.concat(train_Poe, train_simple, axis=1)
    del train_Poe
    del train_simple

    #prepare data for lgb
    cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
    useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]
    features_columns = train_df.columns[~train_df.columns.isin(useless_cols)]

    last_day = datetime.date(2016, 4, 24)
    P_HORIZON = datetime.timedelta(28)
    valid_mask = train_df['date']>str((last_day-P_HORIZON)) #mask for validation set, it is our validation  strategy rn
    
    X_train, y_train = train_df[features_columns], train_df['sales']
    X_valid, y_valid = train_df[valid_mask][features_columns], train_df[valid_mask]['sales']

    train_data = lgb.Dataset(X_train, label= y_train, categorical_feature=cat_feats, free_raw_data=False)
    valid_data = lgb.Dataset(X_valid, label= y_valid, categorical_feature=cat_feats, free_raw_data=False)

    estimator = lgb.train(lgb_params,
                        train_data,
                        valid_sets = [valid_data],
                        verbose_eval = 100,
                        categorical_feature=cat_feats
                        )

    print('Saving model...')
    # save model to file
    model_path = os.path.join(settings.MODEL_DIR, '{0}.{1}.bin'.format(model_name, feature_name))
    pickle.dump(estimator, open(model_path, 'wb'))

    #save validation metrics
    print('Saving Validation Score...')
    y_pred = estimator.predict(train_df[valid_mask][features_columns])
    rmse = sqrt(mean_squared_error(y_valid, y_pred)) #change metrics according to the one you use
    save_score(model_name, feature_name, params= lgb_params, CV_score=rmse)

def predict(feature_name, model_name):

    print('inititaitng prediction dataframe...')
    all_preds = pd.DataFrame() # Create Dummy DataFrame to store predictions

    #load initial train set
    test_Poe = pd.read_feather(os.path.join(settings.FEATURE_DIR, '{0}.tst.feather'.format('Poe')))
    test_simple = pd.read_feather(os.path.join(settings.FEATURE_DIR, '{0}.tst.feather'.format('simple')))
    X_tst = pd.concat(test_Poe, test_simple, axis=1)
    del test_Poe
    del test_simple

    last_day = datetime.date(2016, 4, 24)
    main_time = time.time()

    for PREDICT_DAY in range(1,29):    
        print('Predict | Day:', PREDICT_DAY)
        start_time = time.time()

        model_path = os.path.join(settings.MODEL_DIR, '{0}.{1}.bin'.format(model_name, feature_name))
        estimator = pickle.load(open(model_path, 'rb'))

        grid_df = X_tst.copy()

        useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]
        features_columns = grid_df.columns[~grid_df.columns.isin(useless_cols)]
        day_mask = grid_df['date'] == str((last_day+datetime.timedelta(PREDICT_DAY)))
        X_tst['sales'][day_mask] = estimator.predict(grid_df[day_mask][features_columns])
        
        # Make good column naming and add 
        # to all_preds DataFrame
        temp_df = X_tst[day_mask][['id','sales']]
        temp_df.columns = ['id','F'+str(PREDICT_DAY)]

        if 'id' in list(all_preds):
            all_preds = all_preds.merge(temp_df, on=['id'], how='left')
        else:
            all_preds = temp_df.copy()
            
        print('#'*10, ' %0.2f min round |' % ((time.time() - start_time) / 60),
                    ' %0.2f min total |' % ((time.time() - main_time) / 60),
                    ' %0.2f day sales |' % (temp_df['F'+str(PREDICT_DAY)].sum()))
        del temp_df

    #make submission
    all_preds = all_preds.reset_index(drop=True)
    submission = pd.read_csv(settings.SAMPLE_SUBMISSION)[['id']]
    submission = submission.merge(all_preds, on=['id'], how='left').fillna(0)
    submission.to_csv(os.path.join(settings.SUBMISSION_DIR, '{0}.{1}.sub.csv'.format(model_name, feature_name)))        

if __name__ == "__main__":
    
    feature_name = "Poe+simple" #inspired from https://www.kaggle.com/poedator/m5-under-0-50-optimized
    model_name = 'lgbm2'
    lgb_params = {
                    'boosting_type': 'gbdt',
                    'objective': 'tweedie',
                    'tweedie_variance_power': 1.1,
                    'is_training_metric': True,
                    'metric': 'rmse',
                    'subsample': 0.5,
                    'subsample_freq': 1,
                    'learning_rate': 0.03,
                    'num_leaves': 2**11-1,
                    'min_data_in_leaf': 2**12-1,
                    'feature_fraction': 0.5,
                    'max_bin': 100,
                    'n_estimators': 1400,
                    'boost_from_average': False,
                    'verbose': -1,
                }
    train(feature_name, model_name, lgb_params)
    predict(feature_name, model_name)
    




