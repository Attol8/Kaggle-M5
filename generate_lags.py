from  datetime import datetime, timedelta
import gc
import numpy as np, pandas as pd
import os
import settings 
import logging
from utils import reduce_mem_usage

feature_name = "lags2" #inspired from https://www.kaggle.com/poedator/m5-under-0-50-optimized

def generate_feature(feature_name=feature_name):
    '''creates new features for train and test set then saves them separetely in the correct folder'''

    print('loading raw data')
    train = pd.read_feather(settings.TRAIN_DATA)
    test = pd.read_feather(settings.TEST_DATA)

    trn_tst = train.append(test)
    print('trn_shape:{}, tst_shape: {}, all shape: {}'.format(train.shape, test.shape, trn_tst.shape))

    #create lags features
    lags = [7, 14, 28] 
    lags.extend(range(29, 43))
    lag_cols = [f"lag_{lag}" for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        trn_tst[lag_col] = trn_tst[["id","sales"]].groupby("id")["sales"].shift(lag)

    #create rolling windows mean and std features with various day shift
    for d_shift in [1, 7 , 14, 28, 30, 60]: 
        print('Shifting period:', d_shift)
        for d_window in [7, 14, 28, 30, 60]:
            col_name_m = 'rolling_mean_'+str(d_shift)+'_'+str(d_window)
            trn_tst[col_name_m] = trn_tst.groupby(['id'])["sales"].transform(lambda x: x.shift(d_shift).rolling(d_window).mean()).astype(np.float16)
            col_name_s = 'rolling_std_'+str(d_shift)+'_'+str(d_window)
            trn_tst[col_name_s] = trn_tst.groupby(['id'])["sales"].transform(lambda x: x.shift(d_shift).rolling(d_window).std()).astype(np.float16)

    date_features = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }

    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in trn_tst.columns:
            trn_tst[date_feat_name] = trn_tst[date_feat_name].astype("int16")
        else:
            trn_tst[date_feat_name] = getattr(trn_tst["date"].dt, date_feat_func).astype("int16")

    print(f'shape of all data after lags features: {trn_tst.shape}')    

    with open(os.path.join(settings.FEATURE_DIR, '{0}.fmap'.format(feature_name)), 'w') as f:
        for i, col in enumerate(trn_tst.columns):
            f.write('{}\t{}\tq\n'.format(i, col))

    trn_tst, _ = reduce_mem_usage(trn_tst)
    logging.info('saving features')
    trn_tst[:len(train)].to_feather(os.path.join(settings.FEATURE_DIR, '{0}.trn.feather'.format(feature_name)))
    trn_tst[len(train):].to_feather(os.path.join(settings.FEATURE_DIR, '{0}.tst.feather'.format(feature_name)))

if __name__ == "__main__":
    generate_feature()