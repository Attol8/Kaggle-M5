import datetime
import gc
import numpy as np, pandas as pd
import settings 
import logging
import os, sys, gc, time, warnings, pickle, psutil, random
from math import ceil
from utils import reduce_mem_usage

def generate_feature(feature_name, is_train=True):
    '''creates new features for train and test set then saves them separetely in the correct folder'''

    #load test and train data
    print('Load Main Data')
    if is_train:
        dt = pd.read_feather(settings.TRAIN_DATA)

        #code for taking a sample of the training data (comment if you want full data set)
        last_day = datetime.date(2016, 4, 24)
        P_HORIZON = datetime.timedelta(365)
        sample_mask = dt['date']>str((last_day-P_HORIZON))
        dt = dt[sample_mask]

    else:
        dt = pd.read_feather(settings.TEST_DATA)

    #perform feature engineering
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
    
    # Final list of features
    print(dt.info())

    with open(os.path.join(settings.FEATURE_DIR, '{0}.fmap'.format(feature_name)), 'w') as f:
        for i, col in enumerate(dt.columns):
            f.write('{}\t{}\tq\n'.format(i, col))

    if is_train:
        dt.reset_index().to_feather(os.path.join(settings.FEATURE_DIR, '{0}.trn.feather'.format(feature_name)))
    else:
        dt.reset_index().to_feather(os.path.join(settings.FEATURE_DIR, '{0}.tst.feather'.format(feature_name)))

if __name__ == "__main__":
    generate_feature(feature_name = "best", is_train=False)
    generate_feature(feature_name = "best", is_train=True)

