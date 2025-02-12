from  datetime import datetime, timedelta
import gc
import numpy as np, pandas as pd
import os
import settings 
import logging

feature_name = "Poe" #inspired from https://www.kaggle.com/poedator/m5-under-0-50-optimized

def generate_feature(feature_name=feature_name):
    '''creates new features for train and test set then saves them separetely in the correct folder'''

    logging.info('loading raw data')
    train = pd.read_feather(settings.TRAIN_DATA)
    test = pd.read_feather(settings.TEST_DATA)

    trn_tst = train.append(test)
    logging.info('trn_shape:{}, tst_shape: {}, all shape: {}'.format(train.shape, test.shape, trn_tst.shape))

    logging.info('lag features variables')
    #create lags features
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        trn_tst[lag_col] = trn_tst[["id","sales"]].groupby("id")["sales"].shift(lag)
    logging.info(f'shape of all data after lags features: {trn_tst.shape}')

    #create windows features
    logging.info('windows features variables')
    wins = [7, 28]
    for win in wins :
        for lag,lag_col in zip(lags, lag_cols):
            trn_tst[f"rmean_{lag}_{win}"] = trn_tst[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())
    logging.info(f'shape of all data after lags features: {trn_tst.shape}')

    date_features = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }

    logging.info('dates features variables')
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in trn_tst.columns:
            trn_tst[date_feat_name] = trn_tst[date_feat_name].astype("int16")
        else:
            trn_tst[date_feat_name] = getattr(trn_tst["date"].dt, date_feat_func).astype("int16")

    logging.info(f'shape of all data after lags features: {trn_tst.shape}')    

    with open(os.path.join(settings.FEATURE_DIR, '{0}.fmap'.format(feature_name)), 'w') as f:
        for i, col in enumerate(trn_tst.columns):
            f.write('{}\t{}\tq\n'.format(i, col))

    logging.info('saving features')
    trn_tst[:len(train)].to_feather(os.path.join(settings.FEATURE_DIR, '{0}.trn.feather'.format(feature_name)))
    trn_tst[len(train):].to_feather(os.path.join(settings.FEATURE_DIR, '{0}.tst.feather'.format(feature_name)))

if __name__ == "__main__":
    generate_feature()