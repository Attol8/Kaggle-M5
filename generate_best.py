from  datetime import datetime, timedelta
import gc
import numpy as np, pandas as pd
import settings 
import logging
import os, sys, gc, time, warnings, pickle, psutil, random
from math import ceil

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def generate_feature(feature_name, is_train=True):
    '''creates new features for train and test set then saves them separetely in the correct folder'''
    # Here are reafing all our data 
    # without any limitations and dtype modification

    print('Load Main Data')
    if is_train:
        dt = pd.read_feather(settings.TRAIN_DATA)
    else:
        dt = pd.read_feather(settings.TEST_DATA)

    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)

    wins = [7, 28]
    for win in wins :
        for lag,lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())
    
    date_features = {
        
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
#         "ime": "is_month_end",
#         "ims": "is_month_start",
    }
    
#     dt.drop(["d", "wm_yr_wk", "weekday"], axis=1, inplace = True)
    
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")
    
    #drop useless columns and NA
    useless_cols = ["wm_yr_wk", "weekday"]
    train_cols = dt.columns[~dt.columns.isin(useless_cols)]
    dt = dt[train_cols]
    #if is_train: dt.dropna(inplace = True)
    
    ########################### Final list of features
    #################################################################################
    print(dt.info())

    with open(os.path.join(settings.FEATURE_DIR, '{0}.fmap'.format(feature_name)), 'w') as f:
        for i, col in enumerate(dt.columns):
            f.write('{}\t{}\tq\n'.format(i, col))

    #trn_tst = trn_tst.astype('float32', errors='ignore')
    
    #trn_tst = trn_tst.reset_index(drop=True)
    #save to feathers
    if is_train:
        dt.reset_index().to_feather(os.path.join(settings.FEATURE_DIR, '{0}.trn.feather'.format(feature_name)))
    else:
        dt.reset_index().to_feather(os.path.join(settings.FEATURE_DIR, '{0}.tst.feather'.format(feature_name)))

if __name__ == "__main__":
    generate_feature(feature_name = "best", is_train=False)
    generate_feature(feature_name = "best", is_train=True)

