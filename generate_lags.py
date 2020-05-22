import datetime
import gc
import numpy as np, pandas as pd
import os
import settings 
import logging
from utils import reduce_mem_usage

def generate_feature(feature_name, is_train=True):
    '''creates new features for train and test set then saves them separetely in the correct folder'''

    #load test and train data
    print('Load Main Data')
    if is_train:
        dt = pd.read_feather(settings.TRAIN_DATA)
        #code for taking a sample of the training data (comment if you want fll data set)
        last_day = datetime.date(2016, 4, 24)
        P_HORIZON = datetime.timedelta(365)
        sample_mask = dt['date']>str((last_day-P_HORIZON))
        dt = dt[sample_mask]
    
    else:
        dt = pd.read_feather(settings.TEST_DATA)

    #perform feature engineering

    #create lags features
    lags = [7, 14, 28, 40] 
    lags.extend(range(29, 36))
    lag_cols = [f"lag_{lag}" for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)

    #create rolling windows mean and std features with various day shift
    for d_shift in [7, 14, 28]: 
        print('Shifting period:', d_shift)
        for d_window in [7, 14, 30, 60, 120]:
            col_name_m = 'rmean_'+str(d_shift)+'_'+str(d_window)
            dt[col_name_m] = dt[["id", f"lag_{d_shift}"]].groupby(['id'])[f"lag_{d_shift}"].transform(lambda x: x.rolling(d_window).mean()).astype(np.float16)
            col_name_s = 'smean_'+str(d_shift)+'_'+str(d_window)
            dt[col_name_s] = dt[["id",f"lag_{d_shift}"]].groupby(['id'])[f"lag_{d_shift}"].transform(lambda x: x.rolling(d_window).std()).astype(np.float16)
            col_name_max = 'max_sales_'+str(d_shift)+'_'+str(d_window)
            dt[col_name_max] = dt[["id",f"lag_{d_shift}"]].groupby(['id'])[f"lag_{d_shift}"].transform(lambda x: x.rolling(d_window).max()).astype(np.float16)


    date_features = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }

    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")
  
    # Final list of features
    print(dt.info())
    with open(os.path.join(settings.FEATURE_DIR, '{0}.fmap'.format(feature_name)), 'w') as f:
        for i, col in enumerate(dt.columns):
            f.write('{}\t{}\tq\n'.format(i, col))

    dt, _ = reduce_mem_usage(dt)
    dt = dt.astype('float32', errors='ignore')
    
    if is_train:
        dt.reset_index().to_feather(os.path.join(settings.FEATURE_DIR, '{0}.trn.feather'.format(feature_name)))
    else:
        dt.reset_index().to_feather(os.path.join(settings.FEATURE_DIR, '{0}.tst.feather'.format(feature_name)))
        
if __name__ == "__main__":
    generate_feature(feature_name = "lags3", is_train=True)
    generate_feature(feature_name = "lags3", is_train=False)