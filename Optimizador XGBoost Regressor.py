# Optimizador XGBoost Regressor
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error as MSE 
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
import hyperopt as hp
from hyperopt import tpe, fmin, hp, space_eval, Trials
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample
import xgboost as xgb

######################      BAYESIAN OPTIMIZER FOR DECISION TREE REGRESSOR       #####################

# LOADING THE DATA FROM A .MAT FILE (MATLAB/OCTAVE FILE)
file_name = "data2"
main_path = ("D:\Desktop\Modelo Predictivo PPV")
file_path = (file_name + ".xlsx")
sheet_name = "data2"
dataset = pd.read_excel(main_path + "\\" + file_path, sheet_name)

X=dataset[["Distancia","Carga"]]
y=dataset["PPV"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2,shuffle=True)
X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
y_train.shape, y_test.shape = (-1, 1), (-1, 1)


# DEFINING THE OBJECTIVE FUNCTION

def objective_function(params):

    booster = params['booster']
    eta = params['eta']
    gamma = params['gamma']
    max_depth = int(params['max_depth'])
    n_estimators = int(params['n_estimators'])
    min_child_weight = params['min_child_weight']
    subsample = params['subsample']
    alpha = params['alpha']
    random_state = params['random_state']
    colsample_bytree = params['colsample_bytree']
    colsample_bylevel = params['colsample_bylevel']
    colsample_bynode = params['colsample_bynode']
    reg_lambda = params['reg_lambda']
    grow_policy = params['grow_policy']
    if booster == 'dart':
        sample_type = params['sample_type']
        normalize_type = params['normalize_type']
        rate_drop = params['rate_drop']
        skip_drop = params['skip_drop']


    if booster == 'gbtree':
        model = xgb.XGBRegressor(objective= 'reg:squarederror', booster = booster, eta = eta, gamma = gamma, max_depth = max_depth, n_estimators = n_estimators,
                          min_child_weight = min_child_weight, subsample = subsample, alpha = alpha, random_state = random_state,
                          colsample_bytree = colsample_bytree, colsample_bylevel = colsample_bylevel, grow_policy = grow_policy,
                          colsample_bynode = colsample_bynode, reg_lambda = reg_lambda, n_jobs = -1)
    
    elif booster == 'dart':
        num_round = 50
        model = xgb.XGBRegressor(objective= 'reg:squarederror', booster = booster, eta = eta, gamma = gamma, max_depth = max_depth, n_estimators = n_estimators,
                          min_child_weight = min_child_weight, subsample = subsample, alpha = alpha, random_state = random_state,
                          colsample_bytree = colsample_bytree, sample_type = sample_type, normalize_type = normalize_type,
                          rate_drop = rate_drop, skip_drop = skip_drop, colsample_bylevel = colsample_bylevel, grow_policy = grow_policy,
                          colsample_bynode = colsample_bynode, reg_lambda = reg_lambda, n_jobs = -1)
   
    model.fit(X_train,y_train)
    
    
    
    if booster == "gbtree":
        pred_test = model.predict(X_test)
    elif booster == "dart":
        pred_test = model.predict(X_test, ntree_limit = num_round)
        

    error= MSE(y_test,pred_test)
    r2=-r2_score(y_train,model.predict(X_train))
    
    return float(error)


# DEFINING SEARCH SPACE
search_space = {'booster': hp.choice('booster', ['gbtree',"dart"]),
        'n_estimators': hp.quniform('n_estimators', 50, 3000, 1),
        'eta': hp.uniform('eta', 0, 1),
        'gamma': hp.uniform('gamma', 1, 500),
        'max_depth': hp.quniform('max_depth', 3, 100, 1),
        'min_child_weight': hp.uniform('min_child_weight', 0, 100),
        'random_state': sample(scope.int(hp.quniform('random_state', 4, 8, 1))),
        'subsample': hp.uniform('subsample', 0, 1),
        'alpha': hp.uniform('alpha', 1, 8),
        'colsample_bytree': hp.uniform('colsample_bytree', 0, 1),
        'sample_type': hp.choice('sample_type', ['uniform', 'weighted']),
        'normalize_type': hp.choice('normalize_type', ['tree', 'forest']),
        'grow_policy': hp.choice('grow_policy', ['depthwise', 'lossguide']),
        'rate_drop': hp.uniform('rate_drop', 0, 1),
        'skip_drop': hp.uniform('skip_drop', 0, 1),
        'colsample_bylevel':  hp.uniform('colsample_bylevel', 0, 1),
        'colsample_bynode': hp.uniform('colsample_bynode', 0, 1),
        'reg_lambda':  hp.uniform('reg_lambda', 1, 8)} 

max_evals=300

def select_model(space):
    best_regressor = fmin(fn = objective_function, space = space, algo = tpe.suggest, max_evals = max_evals)
    print(space_eval(space, best_regressor))
 
select_model(search_space)
