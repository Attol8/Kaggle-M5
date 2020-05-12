import pandas as pd
import settings
from utils import save_score, reduce_mem_usage
import os
import datetime
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from math import sqrt
from bayes_opt import BayesianOptimization

#https://medium.com/analytics-vidhya/hyperparameters-optimization-for-lightgbm-catboost-and-xgboost-regressors-using-bayesian-6e7c495947a9

STORE_ID = 0

print('Loading dataset...')
train_df = pd.read_feather(os.path.join(settings.FEATURE_DIR, '{0}.trn.feather'.format('best')))
train_df = train_df[train_df['store_id']==STORE_ID]
train_df, _ = reduce_mem_usage(train_df)       

#prepare data for lgb
cat_feats = ['item_id', 'dept_id', 'cat_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
useless_cols = ['store_id',  'state_id','index', 'id', 'date', "d", "sales"]
features_columns = train_df.columns[~train_df.columns.isin(useless_cols)]
print(f'features columns: {features_columns}')

last_day = datetime.date(2016, 4, 24)
P_HORIZON = datetime.timedelta(28)
valid_mask = train_df['date']>str((last_day-P_HORIZON)) #mask for validation set, it is our validation  strategy rn
train_mask = train_df['date']<str((last_day-2*P_HORIZON)) #introduce gap in training (28 days)
print(max(train_df[train_mask]['date'].unique()))

X_train, y_train = train_df[train_mask][features_columns], train_df[train_mask]['sales']
X_valid, y_valid = train_df[valid_mask][features_columns], train_df[valid_mask]['sales']

def bayesion_opt_lgbm(X_train, y_train, X_valid, y_valid, init_iter=3, n_iters=7, random_state=11, seed = 101, n_estimators = 1300):
    train_data = lgb.Dataset(X_train, label= y_train, categorical_feature=cat_feats, free_raw_data=False)
    valid_data = lgb.Dataset(X_valid, label= y_valid, categorical_feature=cat_feats, free_raw_data=False)

  # Objective Function
    def hyp_lgbm(num_leaves, max_bin, subsample, feature_fraction, bagging_fraction, max_depth, min_split_gain, min_child_weight):
        
        params = {'boosting_type': 'gbdt','n_estimators': n_estimators,
                'objective': 'tweedie', 'subsample_freq': 1, 'is_training_metric': True,
                'metric':'rmse', 'boost_from_average': False, 'learning_rate':0.01, 'min_data_in_leaf': 2**12-1, 'verbose': -1} # Default parameters
        params["num_leaves"] = int(round(num_leaves))
        params['subsample'] = max(min(subsample, 1), 0)
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        #params['min_data_in_leaf'] = int(round(min_data_in_leaf))
        params['max_depth'] = int(round(max_depth))
        params['min_split_gain'] = min_split_gain
        params['max_bin'] = int(round(max_bin))
        params['min_child_weight'] = min_child_weight

        estimator = lgb.train(params,
                            train_data,
                            valid_sets = [valid_data],
                            verbose_eval = 100,
                            categorical_feature=cat_feats
                            )
        y_pred = estimator.predict(X_valid)
        rmse = sqrt(mean_squared_error(y_valid, y_pred))
        return -rmse

  # Domain space-- Range of hyperparameters 
    pds = {'num_leaves': (1500, 3000),
            'max_bin' : (0, 200),
            #'min_data_in_leaf': (1500, 3000),
            'feature_fraction': (0.1, 0.9),
            'bagging_fraction': (0.8, 1),
            'max_depth': (17, 25),
            'min_split_gain': (0.001, 0.1),
            'min_child_weight': (10, 25),
            'subsample' :(0.1, 0.9)
        }

# Surrogate model
    optimizer = BayesianOptimization(hyp_lgbm, pds, random_state=random_state)
                                
# Optimize
    optimizer.maximize(init_points=init_iter, n_iter=n_iters)

bayesion_opt_lgbm(X_train, y_train, X_valid, y_valid, init_iter=5, n_iters=10, random_state=77, seed = 101, n_estimators = 1300)
