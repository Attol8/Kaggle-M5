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

def make_test_train():
    #categories' types
    CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
    "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
    "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
    PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }

    prices = pd.read_csv(settings.PRICES_DATA, dtype=PRICE_DTYPES)
    cal = pd.read_csv(settings.CALENDAR_DATA, dtype=CAL_DTYPES)
    last_day = 1913
    nrows=None
    first_day = 1069 # If you want to load all the data set it to '1' -->  Great  memory overflow  risk ! 1069 is start of 2014 (minus 2.5 years)

    #change data types
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()
            
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()

    numcols = [f"d_{day}" for day in range(first_day,last_day+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})

    #melt the dataframe (tidier data)
    dt = pd.read_csv(settings.MAIN_DATA, 
                        nrows = nrows, usecols = catcols + numcols, dtype = dtype) #load the train dataset

    for col in catcols:
        if col != "id":
            dt[col] = dt[col].cat.codes.astype("int16")
            dt[col] -= dt[col].min()

    #include test data
    for day in range(last_day+1, last_day+ 28 +1):
        dt[f"d_{day}"] = np.nan

    dt = pd.melt(dt,
                    id_vars = catcols,
                    value_vars = [col for col in dt.columns if col.startswith("d_")],
                    var_name = "d",
                    value_name = "sales")

    dt = dt.merge(cal, on= "d", copy = False)
    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
    dt['d'] = [x[2:] for x in dt['d']]
    dt['d'] = dt['d'].astype("int16")
    print(dt.head())
    print('creating test dataframe...')
    train_mask = dt['d'] <= last_day
    test_mask = dt['d'] > (last_day - 100)

    train_dt = dt[train_mask].reset_index(drop=True)
    test_dt = dt[test_mask].reset_index(drop=True)
    train_dt.to_feather(settings.TRAIN_DATA)
    test_dt.to_feather(settings.TEST_DATA)

def save_score(model_name, feature_name, params, CV_score):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    d = {'date and time': dt_string, 'model_name': model_name, 'feature_name': feature_name, 'train_file': '{0}.trn.feather'.format(feature_name), 'metric used': params['metric'],
            'CV score' : CV_score, 'submission': '{0}.{1}.sub.csv'.format(model_name, feature_name)} #make better validation scheme
    
    df = pd.DataFrame(data=d, index=[0])
    df.to_csv(os.path.join(settings.METRIC_DIR, '{0}.{1}.score.csv'.format(model_name, feature_name)))

def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    print(props.columns)
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
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
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist

if __name__ == "__main__":
    #make_out_directories() # run this code to create output directories 
    make_test_train()