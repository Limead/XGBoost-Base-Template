import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

class XgbInput:
    def __init__(self, train_data, target, split_rate = 0.7, gpu = False):
        self.train_data = train_data
        self.target = target
        self.split_rate = split_rate
        self.gpu = gpu
        
    def trainingInput(self):
        xgb_gpu = "gpu_hist" if self.gpu else "auto"
        
        xgb_train = self.train_data.dropna(subset = [f"{self.target}"])

        y_columns = list(f"{self.target}")
                
        x_columns = list(set(xgb_train.columns) - set(y_columns))
        x_columns.sort()
        
        x_data = xgb_train[x_columns]
        y_data = xgb_train[y_columns]
        
        train_x, valid_x, train_y, valid_y = train_test_split(x_data, y_data[f"{self.target}"], test_size = 1 - self.split_rate, random_state = 90)
        
        return({"train_x"   : train_x,
                "train_y"   : train_y,
                "valid_x"   : valid_x,
                "valid_y"   : valid_y,
                "target"    : self.target,
                "xgb_gpu"   : xgb_gpu,
                "x_columns" : x_columns,
                "x_data"    : x_data,
                "y_data"    : y_data[f"{self.target}"]})
    
    def predictionInput(self):
        xgb_gpu = "gpu_hist" if self.gpu else "auto"
        xgb_train = self.train_data.dropna(subset = [f"{self.target}"])
        
        y_columns = list(f"{self.target}")
                
        x_columns = list(set(xgb_train.columns) - set(y_columns))
        x_columns.sort()
        
        x_data = xgb_train[x_columns]
        y_data = xgb_train[y_columns]
        
        y_data = y_data[f"{self.target}"].values
        
        return({"x_data"    : x_data,
                "y_data"    : y_data,
                "xgb_gpu"   : xgb_gpu,
                "x_columns" : x_columns})