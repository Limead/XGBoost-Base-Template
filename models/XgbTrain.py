import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from utils.scorer import RMSE

class XgbTrain:
    def __init__(self, train_x, valid_x, train_y, valid_y, xgb_gpu):
        self.train_x = train_x
        self.valid_x = valid_x
        self.train_y = train_y
        self.valid_y = valid_y
        self.xgb_gpu = xgb_gpu
    
    def xgbTrain(self, n_estimators, learning_rate, max_depth, max_leaves, min_child_weight, subsample, colsample_bytree, max_bin):
        xgb_model = XGBRegressor(booster = "gbtree",
                                 n_estimators = int(n_estimators),
                                 learning_rate = learning_rate,
                                 max_depth = int(max_depth),
                                 max_leaves = int(max_leaves),
                                 min_child_weight = int(min_child_weight),
                                 subsample = subsample,
                                 colsample_bytree = colsample_bytree,
                                 max_bin = int(max_bin),
                                 objective = "reg:squarederror",
                                 tree_method = self.xgb_gpu,
                                 nthread = 6,
                                 seed = 1234)
        
        xgb_model.fit(self.train_x, self.train_y, eval_metric = "rmse", eval_set = [(self.valid_x, self.valid_y)], early_stopping_rounds = 200, verbose = False)
        
        pred = xgb_model.predict(self.valid_x)
        score = -RMSE(pred, self.valid_y)

        return(score)