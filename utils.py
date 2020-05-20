import os
import settings
import pandas as pd
import numpy as np
from datetime import datetime

def make_out_directories(output_path = settings.OUTPUT_DIR):
    if os.path.exists(output_path):
        print(output_path + ' : exists')
    else:
        print('Making Directories')
        os.mkdir(output_path)

    dir_l = ['feature', 'metric', 'model', 'val', 'tst', 'submission']
    for dir in dir_l:
        if os.path.exists(os.path.join(output_path, dir)):
            print(os.path.join(output_path, dir) + ' : exists')
        else:
            os.mkdir(os.path.join(output_path, dir))

def make_test_train(is_train = True, nrows = None, first_day = 1500):
    #categories' types
    CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
    "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
    "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
    PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }
    max_lags = 57
    tr_last = 1913
    prices = pd.read_csv(settings.PRICES_DATA, dtype = PRICE_DTYPES)
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()
            
    cal = pd.read_csv(settings.CALENDAR_DATA, dtype = CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()
    
    start_day = max(1 if is_train  else tr_last-max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    dt = pd.read_csv(settings.MAIN_DATA, 
                     nrows = nrows, usecols = catcols + numcols, dtype = dtype)
    
    for col in catcols:
        if col != "id":
            dt[col] = dt[col].cat.codes.astype("int16")
            dt[col] -= dt[col].min()
    
    if not is_train:
        for day in range(tr_last+1, tr_last+ 28 +1):
            dt[f"d_{day}"] = np.nan
    
    dt = pd.melt(dt,
                  id_vars = catcols,
                  value_vars = [col for col in dt.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales")
    
    print(dt.shape)
    dt = dt.merge(cal, on= "d", copy = False)
    print(dt.shape)
    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
    print(dt.shape)

    if is_train:
        dt.to_feather(settings.TRAIN_DATA)
    else:
        dt.to_feather(settings.TEST_DATA)

def save_score(model_name, feature_name, params, CV_score):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    d = {'date and time': dt_string, 'model_name': model_name, 'feature_name': feature_name, 'train_file': '{0}.trn.feather'.format(feature_name), 'metric used': params['metric'],
            'CV score' : CV_score, 'submission': '{0}.{1}.sub.csv'.format(model_name, feature_name)} #make better validation scheme
    
    df = pd.DataFrame(data=d, index=[0])
    df.to_csv(os.path.join(settings.METRIC_DIR, '{0}.{1}.score.csv'.format(model_name, feature_name)))

def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    #print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        #print(props[col][:5])
        if props[col].dtype not in [object,'datetime64[ns]']:  # Exclude strings
            
            # Print current column type
            #print("******************************")
            #print("Column: ",col)
            #print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
    
    # Print final result
    #print("___MEMORY USAGE AFTER COMPLETION:___")
    #mem_usg = props.memory_usage().sum() / 1024**2 
    #print("Memory usage is: ",mem_usg," MB")
    #print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist

if __name__ == "__main__":
    #make_out_directories() # run this code to create output directories 
    make_test_train(is_train=True)
    make_test_train(False)