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

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed) 

def create_lag_features_for_test(df, day):
    print(f'creating lag and windows feature for day {day}')
    #create lags features
    lags = [7, 14, 28] 
    lags.extend(range(29, 43))
    lag_cols = [f"lag_{lag}" for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        df.loc[df.d == day, lag_col] = \
        df.loc[df.d == day-lag, 'sales'].values

    #create rolling windows mean and std features with various day shift
    windows = [7, 14, 28, 30, 60]
    shifts = [1, 7 , 14, 28, 30, 60, 365]
    for window in windows:
        for shift in shifts:
            df_window = df[(df.d <= day-shift) & (df.d > day-(shift+window))] #filter for date <= day-shift and date > 
            df_window_grouped = df_window.groupby("id").agg({'sales':'mean'}).reindex(df.loc[df.d == day,'id'])
            df.loc[df.d == day,f"rolling_mean_{shift}_{window}"] = \
                df_window_grouped.sales.values
            
            #df_window_grouped = df_window.groupby("id").agg({'sales':'std'}).reindex(df.loc[df.d == day,'id'])
            #df.loc[df.d == day,f"rolling_std_{shift}_{window}"] = \
                #df_window_grouped.sales.values            

    date_features = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }

    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in df.columns:
            df[date_feat_name] = df[date_feat_name].astype("int16")
        else:
            df[date_feat_name] = getattr(df["date"].dt, date_feat_func).astype("int16")
    return df

def save_val_set(feature_name, model_name):
    train_lags = pd.read_feather(os.path.join(settings.FEATURE_DIR, '{0}.trn.feather'.format('lags')))
    train_simple = pd.read_feather(os.path.join(settings.FEATURE_DIR, '{0}.trn.feather'.format('simple'))) 

    print('saving validation set')
    last_day = datetime.date(2016, 4, 24)
    last_day_n = 1913
    P_HORIZON = datetime.timedelta(28)  
    train_lags = train_lags[train_lags['date']>str((last_day-P_HORIZON))]
    train_simple = train_simple[train_simple['d']>(last_day_n-28)]

    train_df = pd.concat([train_lags, train_simple], axis=1)
    train_df = train_df.loc[:,~train_df.columns.duplicated()]
    del train_lags
    del train_simple
    train_df, NAlist = reduce_mem_usage(train_df)

    valid_mask = train_df['date']>str((last_day-P_HORIZON)) #mask for validation set, it is our validation  strategy rn 
    #useless_cols = ['id','state_id','store_id','date','wm_yr_wk', "sales","d", "wm_yr_wk", "weekday"]
    #features_columns = train_df.columns[~train_df.columns.isin(useless_cols)]
    X_val = train_df[valid_mask]

    X_val.to_csv(os.path.join(settings.VAL_DIR, 'val.{0}.{1}.csv'.format(model_name, feature_name)), index=False)

def train(feature_name, model_name, lgb_params):

    for store_id in list(range(10)):   #stores are encoded
        print('loading store {0} dataset'.format(store_id))
        train_lags = pd.read_feather(os.path.join(settings.FEATURE_DIR, '{0}.trn.feather'.format('lags')))
        #train_lags, NAlist = reduce_mem_usage(train_lags)
        train_lags = train_lags[train_lags['store_id']==store_id]
        #train_df = train_all[train_all['store_id']==store_id]

        train_simple = pd.read_feather(os.path.join(settings.FEATURE_DIR, '{0}.trn.feather'.format('simple'))) 
        #train_simple, NAlist = reduce_mem_usage(train_simple)
        train_simple = train_simple[train_simple['store_id']==store_id]
        train_df = pd.concat([train_lags, train_simple], axis=1)
        train_df = train_df.loc[:,~train_df.columns.duplicated()]

        del train_lags
        del train_simple
        train_df, NAlist = reduce_mem_usage(train_df)       

        #prepare data for lgb
        cat_feats = ['item_id', 'dept_id', 'cat_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
        useless_cols = ['id','state_id','store_id','date','wm_yr_wk', "sales","d", "wm_yr_wk", "weekday"]
        features_columns = train_df.columns[~train_df.columns.isin(useless_cols)]

        last_day = datetime.date(2016, 4, 24)
        P_HORIZON = datetime.timedelta(28)
        valid_mask = train_df['date']>str((last_day-P_HORIZON)) #mask for validation set, it is our validation  strategy rn

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
    
    for store_id in list(range(10)):
        store_mask = grid_df['store_id']==store_id
        useless_cols = ['id','state_id','store_id','date','wm_yr_wk', "sales","d", "wm_yr_wk", "weekday"]
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
    test_lags = pd.read_feather(os.path.join(settings.FEATURE_DIR, '{0}.tst.feather'.format('lags')))
    test_simple = pd.read_feather(os.path.join(settings.FEATURE_DIR, '{0}.tst.feather'.format('simple')))

    last_day_n = 1913
    
    X_tst = pd.concat([test_lags, test_simple], axis=1)
    #print(X_tst.head())
    X_tst = X_tst.loc[:,~X_tst.columns.duplicated()]
    X_tst.drop_duplicates(inplace=True)
    del test_lags
    del test_simple
    X_tst.reset_index(drop=True, inplace=True)
    main_time = time.time()

    for PREDICT_DAY in range(1,29):    
        print('Predict | Day:', PREDICT_DAY)
        start_time = time.time()
        print(X_tst.shape)
        grid_df = X_tst.copy()
        day = last_day_n + PREDICT_DAY
        grid_df = create_lag_features_for_test(grid_df, day)
        print(grid_df.shape)
        missing = [x for x in grid_df.columns if x not in X_tst.columns]
        print(missing)
    
        for store_id in list(range(10)):
            model_path = os.path.join(settings.MODEL_DIR, '{0}.{1}.{2}.bin'.format(model_name, feature_name, store_id))
            estimator = pickle.load(open(model_path, 'rb'))
            useless_cols = ['id','state_id','store_id','date','wm_yr_wk', "sales","d", "wm_yr_wk", "weekday"]
            features_columns = grid_df.columns[~grid_df.columns.isin(useless_cols)]
            day_mask = grid_df['d'] == day
            store_mask = grid_df['store_id']==store_id
            mask = day_mask & store_mask
            #print(X_tst_store.columns)
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
                    ' %0.2f day sales |' % (temp_df['F'+str(PREDICT_DAY)].sum()))
        del temp_df

    #make submission
    all_preds = all_preds.reset_index(drop=True)
    submission = pd.read_csv(settings.SAMPLE_SUBMISSION)[['id']]
    submission = submission.merge(all_preds, on=['id'], how='left').fillna(0)
    submission.to_csv(os.path.join(settings.SUBMISSION_DIR, '{0}.{1}.sub.csv'.format(model_name, feature_name)))

if __name__ == "__main__":
    
    feature_name = "lags+simple" #inspired from https://www.kaggle.com/poedator/m5-under-0-50-optimized
    model_name = 'lgbm3'
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
                    'n_estimators': 1300,
                    'boost_from_average': False,
                    'verbose': -1,
                }
    #save_val_set(feature_name, model_name)
    #train(feature_name, model_name, lgb_params)
    #save_metrics(feature_name, model_name)
    predict(feature_name, model_name)
    




