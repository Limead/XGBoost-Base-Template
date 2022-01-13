import os
import sys
from pathlib import Path, PurePath
import pandas as pd
import numpy as np
import argparse
from xgboost import XGBRegressor
from models.XgbInput import XgbInput
from models.XgbTrain import XgbTrain
from models.XgbBayes import XgbBayTrain
from utils.scorer import RMSE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBOOST Modeling Parser")
    # Dataset path 
    parser.add_argument("--data_path", type=str,
                            default=os.path.join(Path(__file__).resolve().parent, 'data/train_data.csv'))
    parser.add_argument("--test_path", type=str,
                            default=os.path.join(Path(__file__).resolve().parent, 'data/test_data.csv'))
    
    # validation data setting
    parser.add_argument("-valid_split_rate", type=float,default=0.7)    
    
    # bayesian opt parameters
    parser.add_argument("-init_points", type=int,default=10)
    parser.add_argument("-n_iter", type=int,default=10)
    
    # target setting
    parser.add_argument("-target", type=str)
    
    # model parameters
    parser.add_argument("-n_estimators", type=int,default=500)
    parser.add_argument("-learning_rate", type=float,default=0.01)
    parser.add_argument("-max_depth", type=int,default=5)
    parser.add_argument("-max_leaves", type=int,default=64)
    parser.add_argument("-max_bin", type=int,default=10)
    parser.add_argument("-min_child_weight", type=int,default=3)
    parser.add_argument("-subsample", type=float,default=1)
    parser.add_argument("-colsample_bytree", type=float,default=1)

    params = vars(parser.parse_known_args()[0])
    data_path = params["data_path"]
    test_path = params["test_path"]

    model_params = {"n_estimators" : (round(params["n_estimators"]/2), params["n_estimators"]*2),
                    "learning_rate" : (params["learning_rate"]/10, params["learning_rate"]*10),
                    "max_depth" : (round(params["max_depth"]/2), params["max_depth"]*2),
                    "max_leaves" : (round(params["max_leaves"]/2), params["max_leaves"]*2),
                    "max_bin" : (round(params["max_bin"]/2), params["max_bin"]*2),
                    "min_child_weight" : (round(params["min_child_weight"]/2), params["min_child_weight"]*2),
                    "subsample" : (params["subsample"], 1),
                    "colsample_bytree" : (params["colsample_bytree"], 1)}

    ########################### data loading ##################################
    train_data = pd.read_csv(f"{data_path}")

    test_data = pd.read_csv(f"{test_path}")

    ########################## XGBOOST Modeling ###############################
    target = params["target"]
    xgIN = XgbInput(train_data, target, params["valid_split_rate"], gpu=True)
    xgIN2 = XgbInput(test_data, target, gpu=True)
    train_set = xgIN.trainingInput()
    test_set = xgIN2.predictionInput()

    train_x = train_set["train_x"]
    train_y = train_set["train_y"]
    valid_x = train_set["valid_x"]
    valid_y = train_set["valid_y"]
    xgb_gpu = train_set["xgb_gpu"]

    xgTR = XgbTrain(train_x ,valid_x ,train_y,valid_y,xgb_gpu)
    xgBT = XgbBayTrain(xgTR.xgbTrain, params["init_points"], params["n_iter"], target, train_x, train_y, valid_x, valid_y, xgb_gpu)

    best_model = xgBT.xgbBayTrain(model_params)

    predict_value = best_model.predict(test_set["x_data"])
    score = RMSE(predict_value, test_set["y_data"])

    print(f'test set : {test_set["y_data"]}')
    print(f'predict value : {predict_value} score : {score}')

