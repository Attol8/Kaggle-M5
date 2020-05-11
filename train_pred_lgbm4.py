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
from utils import save_score, reduce_mem_usage


def create_lag_features_for_test(df, day):
    print(f'creating lag and windows feature for day {day}')
    #create lags features

    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        df.loc[df.d == day, lag_col] = \
        df.loc[df.d == day-lag, 'sales'].values

    wins = [7, 28]
    for window in wins :
        for shift in lags:
            df_window = df[(df.d <= day-shift) & (df.d > day-(shift+window))] #filter for date <= day-shift and date > 
            df_window_grouped = df_window.groupby("id").agg({'sales':'mean'}).reindex(df.loc[df.d == day,'id'])
            df.loc[df.d == day,f"rmean_{shift}_{window}"] = \
                df_window_grouped.sales.values
    return df

def save_val_set(feature_name, model_name):
    train_df = pd.read_feather(os.path.join(settings.FEATURE_DIR, '{0}.trn.feather'.format('best')))

    print('saving validation set')
    last_day = datetime.date(2016, 4, 24)
    #last_day_n = 1913
    P_HORIZON = datetime.timedelta(28)  
    valid_mask = train_df['date']>str((last_day-P_HORIZON)) #mask for validation set, it is our validation  strategy rn 

    X_val = train_df[valid_mask]
    X_val.to_csv(os.path.join(settings.VAL_DIR, 'val.{0}.{1}.csv'.format(model_name, feature_name)), index=False)

def train(feature_name, model_name, lgb_params):

    for store_id in list(range(10)):   #stores are encoded
        print('\n')
        print('loading store {0} dataset'.format(store_id))
        train_df = pd.read_feather(os.path.join(settings.FEATURE_DIR, '{0}.trn.feather'.format('best')))
        train_df = train_df[train_df['store_id']==store_id]
        train_df, _ = reduce_mem_usage(train_df)       

        #prepare data for lgb
        cat_feats = ['item_id', 'dept_id', 'cat_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
        useless_cols = ['store_id',  'state_id','index', 'id', 'date', "d", "sales"]
        features_columns = train_df.columns[~train_df.columns.isin(useless_cols)]
        print(f'features columns: {features_columns}')

        last_day = datetime.date(2016, 4, 24)
        P_HORIZON = datetime.timedelta(28)
        valid_mask = train_df['date']>str((last_day-P_HORIZON)) #mask for validation set, it is our validation  strategy rn
        print(max(train_df[valid_mask]['date'].unique()))
        X_train, y_train = train_df[features_columns], train_df['sales']
        X_valid, y_valid = train_df[valid_mask][features_columns], train_df[valid_mask]['sales']
        
        del train_df; gc.collect()

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
        model_path = os.path.join(settings.MODEL_DIR, '{0}.{1}.{2}.bin'.format(model_name, feature_name, store_id))
        pickle.dump(estimator, open(model_path, 'wb'))

def save_metrics(feature_name, model_name):
    #save validation metric
    print('saving validation metric')
    valid_data = pd.read_csv(os.path.join(settings.VAL_DIR, 'val.{0}.{1}.csv'.format(model_name, feature_name)))
    valid_data.reset_index(drop=True, inplace=True)
    y_valid = valid_data['sales']

    grid_df = valid_data.copy()
    grid_df['sales'] = 0
    print(grid_df.shape)
    
    for store_id in list(range(10)):
        store_mask = grid_df['store_id']==store_id
        useless_cols = ['store_id',  'state_id','index', 'id', 'date', "d", "sales"]
        features_columns = grid_df.columns[~grid_df.columns.isin(useless_cols)]

        model_path = os.path.join(settings.MODEL_DIR, '{0}.{1}.{2}.bin'.format(model_name, feature_name, store_id))
        estimator = pickle.load(open(model_path, 'rb'))

        grid_df.loc[store_mask, 'sales'] = estimator.predict(grid_df[store_mask][features_columns])
        y_pred_store = grid_df.loc[store_mask, 'sales']
        y_valid_store = valid_data.loc[valid_data['store_id']==store_id, 'sales']
        #y_valid = y_valid.values #convert to np array as y_pred is np array
        rmse = sqrt(mean_squared_error(y_valid_store, y_pred_store))
        print('store {1} validation rmse is : {0}'.format(rmse, store_id))

    y_valid, y_pred = valid_data['sales'], grid_df['sales']
    rmse = sqrt(mean_squared_error(y_valid, y_pred)) #change metrics according to the one you use
    print('validation rmse is : {0}'.format(rmse))
    save_score(model_name, feature_name, params= lgb_params, CV_score=rmse)  

    with open(os.path.join(settings.METRIC_DIR, '{0}.{1}.fmap'.format(model_name, feature_name)), 'w') as f:
        for i, col in enumerate(features_columns):
            f.write('{}\t{}\tq\n'.format(i, col))

def predict(feature_name, model_name):

    print('initiating prediction dataframe...')
    all_preds = pd.DataFrame() # Create Dummy DataFrame to store predictions

    #load initial test set
    X_tst = pd.read_feather(os.path.join(settings.FEATURE_DIR, '{0}.tst.feather'.format('best')))
    X_tst['d'] = X_tst['d'].str[-4:]
    X_tst['d'] = X_tst['d'].astype('float32')
    last_day_n = 1913
    #X_tst.reset_index(drop=True, inplace=True)
    print(f'X_tst shape: {X_tst.shape}')
    print(X_tst.columns)
    main_time = time.time()

    for PREDICT_DAY in range(1,29):    
        print('Predict | Day:', PREDICT_DAY)
        start_time = time.time()
        grid_df = X_tst.copy()
        day = last_day_n + PREDICT_DAY
        grid_df = create_lag_features_for_test(grid_df, day)
        print(f'missing columns {[x for x in grid_df.columns if x not in X_tst.columns]}')
        print(f'missing values total {grid_df.isnull().sum().sum()}')
        #print(f'columns with missing values: {grid_df.columns[grid_df.isnull().any()].tolist()}')

        for store_id in list(range(10)):
            model_path = os.path.join(settings.MODEL_DIR, '{0}.{1}.{2}.bin'.format(model_name, feature_name, store_id))
            estimator = pickle.load(open(model_path, 'rb'))
            useless_cols = ['store_id',  'state_id','index', 'id', 'date', "d", "sales"]
            features_columns = grid_df.columns[~grid_df.columns.isin(useless_cols)]
            day_mask = X_tst['d'] == day
            store_mask = X_tst['store_id']==store_id
            mask = (day_mask) & (store_mask)
            #print(X_tst_store.columns)
            #print(grid_df[mask][features_columns].head())
            X_tst.loc[mask, 'sales'] = estimator.predict(grid_df[mask][features_columns])
        
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
                    ' %0.2f day sales |' % (temp_df['F'+str(PREDICT_DAY)].sum()),
                    '\n')
        del temp_df

    #make submission
    all_preds = all_preds.reset_index(drop=True)
    submission = pd.read_csv(settings.SAMPLE_SUBMISSION)[['id']]
    submission = submission.merge(all_preds, on=['id'], how='left').fillna(0)
    submission.to_csv(os.path.join(settings.SUBMISSION_DIR, '{0}.{1}.sub.csv'.format(model_name, feature_name)), index=False)

if __name__ == "__main__":
    
    feature_name = "best" #inspired from https://www.kaggle.com/poedator/m5-under-0-50-optimized
    model_name = 'lgbm3'
    lgb_params ={
        "objective" : "poisson",
        "metric" :"rmse",
        "force_row_wise" : True,
        "learning_rate" : 0.075,
#         "sub_feature" : 0.8,
        "sub_row" : 0.75,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
#         "nthread" : 4
        "metric": ["rmse"],
    'verbosity': 1,
    'num_iterations' : 1200,
    'num_leaves': 128,
    "min_data_in_leaf": 100,
}

    #save_val_set(feature_name, model_name)
    #train(feature_name, model_name, lgb_params)
    #save_metrics(feature_name, model_name)
    predict(feature_name, model_name)
    




