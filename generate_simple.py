import datetime
import gc
import numpy as np, pandas as pd
import settings 
import logging
import os, sys, gc, time, warnings, pickle, psutil, random
from math import ceil
from utils import reduce_mem_usage

## Merging by concat to not lose dtypes
def merge_by_concat(df1, df2, merge_on, release= True):
    merged_gf = df1[merge_on]
    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    return df1

def generate_feature(feature_name, is_train = True):
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

    CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
    "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
    "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32'}
    PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }
    prices_df = pd.read_csv(settings.PRICES_DATA, dtype=PRICE_DTYPES)
    calendar_df = pd.read_csv(settings.CALENDAR_DATA, dtype =CAL_DTYPES)
    dt = dt.reset_index(drop=True)

    #change data types 
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices_df[col] = prices_df[col].cat.codes.astype("int16")
            prices_df[col] -= prices_df[col].min()
            
    calendar_df["date"] = pd.to_datetime(calendar_df["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            calendar_df[col] = calendar_df[col].cat.codes.astype("int16")
            calendar_df[col] -= calendar_df[col].min()

    ########################### Product Release date
    #################################################################################
    print('Release week')

    # It seems that leadings zero values
    # in each train_df item row
    # are not real 0 sales but mean
    # absence for the item in the store
    # we can safe some memory by removing
    # such zeros

    # Prices are set by week
    # so it we will have not very accurate release week 
    release_df = prices_df.groupby(['store_id','item_id'])['wm_yr_wk'].agg(['min']).reset_index()
    release_df.columns = ['store_id','item_id','release']

    # Now we can merge release_df
    dt = merge_by_concat(dt, release_df, ['store_id','item_id'])
    del release_df

    # We want to remove some "zeros" rows
    # from dt 
    # to do it we need wm_yr_wk column
    # let's merge partly calendar_df to have it
    dt = merge_by_concat(dt, calendar_df[['wm_yr_wk','d']], ['d'])
    print('making release')                    
    # Now we can cutoff some rows 
    # and safe memory 
    #dt = dt[dt['wm_yr_wk']>=dt['release']]
    dt = dt.reset_index(drop=True)

    # Should we keep release week 
    # as one of the features?
    # Only good CV can give the answer.
    # Let's minify the release values.
    # Min transformation will not help here 
    # as int16 -> Integer (-32768 to 32767)
    # and our dt['release'].max() serves for int16
    # but we have have an idea how to transform 
    # other columns in case we will need it
    #dt['release'] = dt['release'] - dt['release'].min()
    #print(dt.isna().sum())
    dt['release'] = dt['release'].astype(np.int16)

    ########################### Prices
    #################################################################################
    print('Prices')

    # We can do some basic aggregations
    prices_df['price_max'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('max')
    prices_df['price_min'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('min')
    prices_df['price_std'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('std')
    prices_df['price_mean'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('mean')

    # and do price normalization (min/max scaling)
    prices_df['price_norm'] = prices_df['sell_price']/prices_df['price_max']
    prices_df.drop('price_max', axis=1, inplace=True)
    # Some items are can be inflation dependent
    # and some items are very "stable"
    prices_df['price_nunique'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('nunique')
    prices_df['item_nunique'] = prices_df.groupby(['store_id','sell_price'])['item_id'].transform('nunique')

    # I would like some "rolling" aggregations
    # but would like months and years as "window"
    calendar_prices = calendar_df[['wm_yr_wk','month','year']]
    calendar_prices = calendar_prices.drop_duplicates(subset=['wm_yr_wk'])
    prices_df = prices_df.merge(calendar_prices[['wm_yr_wk','month','year']], on=['wm_yr_wk'], how='left')
    del calendar_prices

    # Now we can add price "momentum" (some sort of)
    # Shifted by week 
    # by month mean
    # by year mean
    prices_df['price_momentum'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id'])['sell_price'].transform(lambda x: x.shift(1))
    prices_df['price_momentum_m'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id','month'])['sell_price'].transform('mean')
    prices_df['price_momentum_y'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id','year'])['sell_price'].transform('mean')

    del prices_df['month'], prices_df['year']

    ########################### Merge prices and save part 2
    #################################################################################
    print('Merge prices')

    # Merge Prices
    original_columns = list(dt)
    dt = dt.merge(prices_df, on=['store_id','item_id','wm_yr_wk'], how='left')
    #keep_columns = [col for col in list(dt) if col not in original_columns]
    #dt = dt[MAIN_INDEX+keep_columns]
    dt, _ = reduce_mem_usage(dt)

    # Safe part 2
    #dt.to_pickle('grid_part_2.pkl')
    #print('Size:', dt.shape)

    # We don't need prices_df anymore
    del prices_df

    ########################### Merge calendar
    #################################################################################
    #dt = dt[MAIN_INDEX]

    # Merge calendar partly
    icols = ['date',
            'd',
            'event_name_1',
            'event_type_1',
            'event_name_2',
            'event_type_2',
            'snap_CA',
            'snap_TX',
            'snap_WI']

    dt = dt.merge(calendar_df[icols], on=['d'], how='left')

    # Minify data
    # 'snap_' columns we can convert to bool or int8
    icols = ['event_name_1',
            'event_type_1',
            'event_name_2',
            'event_type_2',
            'snap_CA',
            'snap_TX',
            'snap_WI']
    for col in icols:
        dt[col] = dt[col].astype('category')

    # Convert to DateTime
    dt['date'] = pd.to_datetime(dt['date'])

    # Make some features from date
    dt['tm_d'] = dt['date'].dt.day.astype(np.int8)
    dt['tm_w'] = dt['date'].dt.week.astype(np.int8)
    dt['tm_m'] = dt['date'].dt.month.astype(np.int8)
    dt['tm_y'] = dt['date'].dt.year
    dt['tm_y'] = (dt['tm_y'] - dt['tm_y'].min()).astype(np.int8)
    dt['tm_wm'] = dt['tm_d'].apply(lambda x: ceil(x/7)).astype(np.int8)

    dt['tm_dw'] = dt['date'].dt.dayofweek.astype(np.int8)
    dt['tm_w_end'] = (dt['tm_dw']>=5).astype(np.int8)

    # Remove date
    del dt['date']

    # We don't need calendar_df anymore
    del calendar_df

    # Convert 'd' to int
    dt['d'] = dt['d'].apply(lambda x: x[2:]).astype(np.int16)
    dt = dt.astype('float32', errors='ignore')
    
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
    generate_feature(feature_name = "simple", is_train=True)
    generate_feature(feature_name = "simple", is_train=False)

