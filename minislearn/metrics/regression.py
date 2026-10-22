import numpy as np 



def mean_absolute_error(y_true, y_hat):
    return np.mean(np.abs(y_true - y_hat))

def mean_squared_error(y_true, y_hat):
    return np.mean((y_true - y_hat)**2)

def root_mean_squared_error(y_true, y_hat):
    return np.sqrt(mean_squared_error(y_true, y_hat)) 

def r2_score(y_true, y_hat):
    y_mean = np.mean(y_true)

    ss_res = np.sum((y_true - y_hat) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)

    return 1 - (ss_res / ss_tot)


