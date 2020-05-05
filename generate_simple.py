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

## Simple "Memory profilers" to see memory usage
def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 
        
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

## Merging by concat to not lose dtypes
def merge_by_concat(df1, df2, merge_on, release= True):
    merged_gf = df1[merge_on]
    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    return df1

def generate_feature(feature_name):
    '''creates new features for train and test set then saves them separetely in the correct folder'''
    # Here are reafing all our data 
    # without any limitations and dtype modification

    print('Load Main Data')
    CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
    "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
    "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32'}
    PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }
    train = pd.read_feather(settings.TRAIN_DATA)
    test = pd.read_feather(settings.TEST_DATA)
    trn_tst = train.append(test)[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'd',
    'sales']]
    trn_tst['d'] = 'd_' + trn_tst['d'].astype(str)
    train_len = len(train) #we need it as we are going to delete train set for memory reason
    print(len(trn_tst))

    del train
    del test
    gc.collect()

    #train_df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
    prices_df = pd.read_csv(settings.PRICES_DATA, dtype=PRICE_DTYPES)
    calendar_df = pd.read_csv(settings.CALENDAR_DATA, dtype =CAL_DTYPES)
    trn_tst = trn_tst.reset_index(drop=True)

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
    trn_tst = merge_by_concat(trn_tst, release_df, ['store_id','item_id'])
    del release_df

    # We want to remove some "zeros" rows
    # from trn_tst 
    # to do it we need wm_yr_wk column
    # let's merge partly calendar_df to have it
    trn_tst = merge_by_concat(trn_tst, calendar_df[['wm_yr_wk','d']], ['d'])
    print('making release')                    
    # Now we can cutoff some rows 
    # and safe memory 
    #trn_tst = trn_tst[trn_tst['wm_yr_wk']>=trn_tst['release']]
    trn_tst = trn_tst.reset_index(drop=True)

    # Let's check our memory usage
    print("{:>20}: {:>8}".format('Original trn_tst',sizeof_fmt(trn_tst.memory_usage(index=True).sum())))

    # Should we keep release week 
    # as one of the features?
    # Only good CV can give the answer.
    # Let's minify the release values.
    # Min transformation will not help here 
    # as int16 -> Integer (-32768 to 32767)
    # and our trn_tst['release'].max() serves for int16
    # but we have have an idea how to transform 
    # other columns in case we will need it
    #trn_tst['release'] = trn_tst['release'] - trn_tst['release'].min()
    #print(trn_tst.isna().sum())
    trn_tst['release'] = trn_tst['release'].astype(np.int16)

    # Let's check again memory usage
    print("{:>20}: {:>8}".format('Reduced trn_tst',sizeof_fmt(trn_tst.memory_usage(index=True).sum())))

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
    original_columns = list(trn_tst)
    trn_tst = trn_tst.merge(prices_df, on=['store_id','item_id','wm_yr_wk'], how='left')
    #keep_columns = [col for col in list(trn_tst) if col not in original_columns]
    #trn_tst = trn_tst[MAIN_INDEX+keep_columns]
    trn_tst = reduce_mem_usage(trn_tst)

    # Safe part 2
    #trn_tst.to_pickle('grid_part_2.pkl')
    #print('Size:', trn_tst.shape)

    # We don't need prices_df anymore
    del prices_df

    ########################### Merge calendar
    #################################################################################
    #trn_tst = trn_tst[MAIN_INDEX]

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

    trn_tst = trn_tst.merge(calendar_df[icols], on=['d'], how='left')

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
        trn_tst[col] = trn_tst[col].astype('category')

    # Convert to DateTime
    trn_tst['date'] = pd.to_datetime(trn_tst['date'])

    # Make some features from date
    trn_tst['tm_d'] = trn_tst['date'].dt.day.astype(np.int8)
    trn_tst['tm_w'] = trn_tst['date'].dt.week.astype(np.int8)
    trn_tst['tm_m'] = trn_tst['date'].dt.month.astype(np.int8)
    trn_tst['tm_y'] = trn_tst['date'].dt.year
    trn_tst['tm_y'] = (trn_tst['tm_y'] - trn_tst['tm_y'].min()).astype(np.int8)
    trn_tst['tm_wm'] = trn_tst['tm_d'].apply(lambda x: ceil(x/7)).astype(np.int8)

    trn_tst['tm_dw'] = trn_tst['date'].dt.dayofweek.astype(np.int8)
    trn_tst['tm_w_end'] = (trn_tst['tm_dw']>=5).astype(np.int8)

    # Remove date
    del trn_tst['date']

    # We don't need calendar_df anymore
    del calendar_df

    # Convert 'd' to int
    trn_tst['d'] = trn_tst['d'].apply(lambda x: x[2:]).astype(np.int16)

    ########################### Summary
    #################################################################################
                        
    # Let's check again memory usage
    print("{:>20}: {:>8}".format('Full Grid',sizeof_fmt(trn_tst.memory_usage(index=True).sum())))
    print('Size:', trn_tst.shape)
    
    ########################### Final list of features
    #################################################################################
    print(trn_tst.info())

    with open(os.path.join(settings.FEATURE_DIR, '{0}.fmap'.format(feature_name)), 'w') as f:
        for i, col in enumerate(trn_tst.columns):
            f.write('{}\t{}\tq\n'.format(i, col))

    logging.info('saving features')
    trn_tst = trn_tst.astype('float32', errors='ignore')
    
    #trn_tst = trn_tst.reset_index(drop=True)
    #save to feathers
    trn_tst[:train_len].reset_index().to_feather(os.path.join(settings.FEATURE_DIR, '{0}.trn.feather'.format(feature_name)))
    trn_tst[train_len:].reset_index().to_feather(os.path.join(settings.FEATURE_DIR, '{0}.tst.feather'.format(feature_name)))

if __name__ == "__main__":
    generate_feature(feature_name = "simple" )

