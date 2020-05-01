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

if __name__ == "__main__":
    #make_out_directories() # run this code to create output directories 
    make_test_train()