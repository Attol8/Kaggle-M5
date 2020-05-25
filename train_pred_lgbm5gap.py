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
from math import sqrt, floor
import math
from utils import save_score, reduce_mem_usage


def create_lag_features_for_test(df, day):
    print(f'creating lag and windows feature for day {day}')
    #create lags features

    lags = [7, 14, 28, 40] 
    lags.extend(range(29, 36))
    lag_cols = [f"lag_{lag}" for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        df.loc[df.d == day, lag_col] = \
        df.loc[df.d == day-lag, 'sales'].values

    wins = [7, 14, 30, 60, 120]
    for window in wins :
        for shift in [7, 14, 28]:
            df_window = df[(df.d <= day-shift) & (df.d > day-(shift+window))] #filter for date <= day-shift and date > day-(shift+window)
            #mean
            df_window_grouped = df_window.groupby("id").agg({'sales':'mean'}).reindex(df.loc[df.d == day,'id'])
            df.loc[df.d == day,f"rmean_{shift}_{window}"] = \
                df_window_grouped.sales.values
            #std
            df_window_grouped = df_window.groupby("id").agg({'sales':'std'}).reindex(df.loc[df.d == day,'id'])
            df.loc[df.d == day,f"smean_{shift}_{window}"] = \
                df_window_grouped.sales.values
            #max sales
            df_window_grouped = df_window.groupby("id").agg({'sales':'max'}).reindex(df.loc[df.d == day,'id'])
            df.loc[df.d == day,f"max_sales_{shift}_{window}"] = \
                df_window_grouped.sales.values

    df_window = df[(df.d <= day-7) & (df.d > day-(7+28))] #filter for date <= day-shift and date > day-(shift+window)
    #mean
    df_window_grouped = df_window.groupby("id").agg({'sales':'mean'}).reindex(df.loc[df.d == day,'id'])
    df.loc[df.d == day,f"rmean_{7}_{28}"] = \
        df_window_grouped.sales.values
    return df

def numbers_check(features_l):
    df_l =[]
    for feature in features_l:
        df_curr = pd.read_feather(os.path.join(settings.FEATURE_DIR, '{0}.trn.feather'.format(feature)))
        df_l.append(df_curr)
        print(f'feat shape is : {df_curr.shape}')

    df = pd.concat(df_l, axis=1)
    df = df.loc[:,~df.columns.duplicated()]
    print(f'train shape is : {df.shape}')

def join_features(features_l, store_id, is_train=True):
    if is_train:
        df_l =[]
        for feature in features_l:
            if feature == 'lags3':
                df_curr =  pd.read_feather(os.path.join(settings.FEATURE_DIR, '{0}.{1}.trn.feather'.format(feature, store_id)))
                df_curr.reset_index(drop=True, inplace=True)
                df_l.append(df_curr)
            else:
                df_curr = pd.read_feather(os.path.join(settings.FEATURE_DIR, '{0}.trn.feather'.format(feature)))
                df_curr, _ = reduce_mem_usage(df_curr)
                df_curr = df_curr[df_curr['store_id'] == store_id]
                df_curr.reset_index(drop=True, inplace=True)
                df_l.append(df_curr)


    else:
        df_l =[]
        for feature in features_l:
            df_curr = pd.read_feather(os.path.join(settings.FEATURE_DIR, '{0}.tst.feather'.format(feature)))
            df_l.append(df_curr)

    df = pd.concat(df_l, axis=1) 
    df = df.loc[:,~df.columns.duplicated()]
    print(f'df final shape {df.shape}')
    #if is_train: df.dropna(inplace = True)

    return df

def save_val_set(feature_name, model_name, features_l):

    val_l = []
    #get validation set
    for store_id in list(range(10)):
        train_df = join_features(features_l, store_id)
        last_day = datetime.date(2016, 4, 24)
        P_HORIZON = datetime.timedelta(28)
        valid_mask = train_df['date']>str((last_day-P_HORIZON)) #mask for validation set, it is our validation strategy rn 
        X_val_store = train_df[valid_mask]
        print(f'columns with nas {X_val_store.columns[X_val_store.isna().any()].tolist()}')
        #print(X_val_store.head())
        val_l.append(X_val_store)
    
    X_val = pd.concat(val_l, axis=0)
    print('saving validation set')
    print(f'validation set shape is : {X_val.shape}')
    X_val.to_csv(os.path.join(settings.VAL_DIR, 'val.{0}.{1}.csv'.format(model_name, feature_name)), index=False)

def train(feature_name, model_name, lgb_params, features_l, features_selection=True):

    for store_id in list(range(10)):   #stores are encoded
        print('\n')
        print('loading store {0} dataset'.format(store_id))
        train_df = join_features(features_l, store_id=store_id)
        train_df, _ = reduce_mem_usage(train_df) 

        #prepare data for lgb
        cat_feats = ['item_id', 'dept_id', 'cat_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
        useless_cols = ['store_id',  'wm_yr_wk', 'state_id','index', 'id', 'date', "d", "sales"]
        features_columns = train_df.columns[~train_df.columns.isin(useless_cols)]
        
        if features_selection:
            last_day = datetime.date(2016, 4, 24)
            P_HORIZON = datetime.timedelta(28)
            valid_mask = train_df['date']>str((last_day-P_HORIZON)) #mask for validation set, it is our validation  strategy rn
            train_mask = train_df['date']<str((last_day-2*P_HORIZON)) #introduce gap in training (28 days)
            X_train, y_train = train_df[train_mask][features_columns], train_df[train_mask]['sales']
            X_valid, y_valid = train_df[valid_mask][features_columns], train_df[valid_mask]['sales']        

        else:
            np.random.seed(777)
            idx_len = math.floor(0.2 * train_df.shape[0])
            print(idx_len)
            fake_valid_inds = np.random.choice(train_df.index.values, idx_len , replace = False)
            train_inds = np.setdiff1d(train_df.index.values, fake_valid_inds)
            X_train, y_train = train_df.loc[train_inds][features_columns], train_df.loc[train_inds]['sales']
            X_valid, y_valid = train_df.loc[fake_valid_inds][features_columns], train_df.loc[fake_valid_inds]['sales']
            print(f'validation set shape is : {X_valid.shape}')
   
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
        useless_cols = ['store_id', 'wm_yr_wk', 'state_id','index', 'id', 'date', 'd', 'sales']
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
    print(f'features columns: {features_columns}')
    save_score(model_name, feature_name, params= lgb_params, CV_score=rmse)  

    #check for features importance
    feature_importance(base_rmse= rmse, y_valid= y_valid, valid_data=valid_data, features =features_columns)
    with open(os.path.join(settings.METRIC_DIR, '{0}.{1}.fmap'.format(model_name, feature_name)), 'w') as f:
        for i, col in enumerate(features_columns):
            f.write('{}\t{}\tq\n'.format(i, col))

def feature_importance(base_rmse, y_valid, valid_data, features):

    #set all sales on temporary df to 0
    for col in features:
        grid_df = valid_data.copy()
        grid_df['sales'] = 0

        # Error here appears if we have "categorical" features and can't 
        # do np.random.permutation without disrupt categories
        # so we need to check if feature is numerical
        if grid_df[col].dtypes.name != 'category':
            grid_df[col] = np.random.permutation(grid_df[col].values)
            #grid_df['preds'] = test_model.predict(grid_df[features_columns])

        #load model and make predictions on grid_df for each store
        for store_id in list(range(10)):
            store_mask = grid_df['store_id']==store_id
            #useless_cols = ['store_id', 'wm_yr_wk', 'state_id','index', 'id', 'date', 'd', 'sales']
            #features_columns = grid_df.columns[~grid_df.columns.isin(useless_cols)]

            model_path = os.path.join(settings.MODEL_DIR, '{0}.{1}.{2}.bin'.format(model_name, feature_name, store_id))
            estimator = pickle.load(open(model_path, 'rb'))

            grid_df.loc[store_mask, 'sales'] = estimator.predict(grid_df[store_mask][features])

        y_valid, y_pred = valid_data['sales'], grid_df['sales']
        cur_score = sqrt(mean_squared_error(y_valid, y_pred)) #change metrics according to the one you use
        
        # If our current rmse score is less than base score
        # it means that feature most probably is a bad one
        # and our model is learning on noise
        print(col, np.round(cur_score - base_rmse, 4))


def predict(feature_name, model_name, features_l):

    print('initiating prediction dataframe...')
    all_preds = pd.DataFrame() # Create Dummy DataFrame to store predictions

    #load initial test set
    X_tst = join_features(features_l, 0, is_train=False) #0 because we are merging the whole dataset not by store
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
        print(f'columns with nas {grid_df[grid_df.d == day].columns[grid_df[grid_df.d == day].isna().any()].tolist()}')
        #print(f'columns with missing values: {grid_df.columns[grid_df.isnull().any()].tolist()}')

        for store_id in list(range(10)):
            model_path = os.path.join(settings.MODEL_DIR, '{0}.{1}.{2}.bin'.format(model_name, feature_name, store_id))
            estimator = pickle.load(open(model_path, 'rb'))
            useless_cols = ['store_id', 'wm_yr_wk', 'state_id','index', 'id', 'date', 'd', 'sales']
            #cat_feats = ['item_id', 'dept_id', 'cat_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
            features_columns = grid_df.columns[~grid_df.columns.isin(useless_cols)]
            day_mask = grid_df['d'] == day
            store_mask = grid_df['store_id']==store_id
            mask = (day_mask) & (store_mask)
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
    
    feature_name = "best+simple+lags" 
    model_name = 'lgbm5gap'
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

    features_l = ['best', 'simple', 'lags3']
    #numbers_check(features_l)
    #save_val_set(feature_name, model_name, features_l=features_l)
    train(feature_name, model_name, lgb_params, features_l=features_l, features_selection=False)
    #save_metrics(feature_name, model_name) #uncomment when running features tests
    predict(feature_name, model_name, features_l=features_l)
    




