import os
import numpy as np
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization

class XgbBayTrain:
    def __init__(self, xgb_train_func, init_points, n_iter, target, train_x, train_y, valid_x, valid_y, xgb_gpu):
        self.xgb_train_func = xgb_train_func
        self.init_points = init_points
        self.n_iter = n_iter
        self.target = target
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.xgb_gpu = xgb_gpu
    
    def xgbBayTrain(self, params):
        
        OPT_Res = BayesianOptimization(self.xgb_train_func, params, verbose = 2)
        OPT_Res.maximize(init_points = self.init_points,
                         n_iter = self.n_iter)
        
        new_xgb_model = XGBRegressor(booster = "gbtree",
                                    n_estimators = int(OPT_Res.max["params"]["n_estimators"]),
                                    learning_rate = OPT_Res.max["params"]["learning_rate"],
                                    max_depth = int(OPT_Res.max["params"]["max_depth"]),
                                    max_leaves = int(OPT_Res.max["params"]["max_leaves"]),
                                    max_bin = int(OPT_Res.max["params"]["max_bin"]),
                                    min_child_weight = int(OPT_Res.max["params"]["min_child_weight"]),
                                    subsample = OPT_Res.max["params"]["subsample"],
                                    colsample_bytree = OPT_Res.max["params"]["colsample_bytree"],
                                    objective = "reg:squarederror",
                                    tree_method = self.xgb_gpu,
                                    nthread = 6)

        new_xgb_model.fit(self.train_x, self.train_y, eval_metric = "rmse", eval_set = [(self.valid_x, self.valid_y)], early_stopping_rounds = 200, verbose = False)

        return(new_xgb_model)