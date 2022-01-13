from math import sqrt
import numpy as np

def RMSE(pred, obs):
    return(sqrt(np.mean((pred - obs) ** 2)))