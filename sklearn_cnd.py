from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression

import numpy as np
from copy import copy
import random
from sklearn.metrics import mean_squared_error
mse = mean_squared_error


#@DeprecationWarning("not_essay")
def train_baye(X_train, y_train):
    reg = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)
    reg.fit(X_train, y_train)
    return reg

def generate(attrs, tb, IC, A, params):
    attrs_excA = list(filter(lambda x: x != A, attrs))
    x_train, y_train = [], []
    X_train = []
    if params['func_name'] == "linear":
        # use all attrs
        x_train = np.array([[tb[x][i] for i in attrs_excA] for x in IC])
        y_train = np.array([tb[x][A] for x in IC])
        return [x_train], y_train, [x_train]
    elif params['func_name'] == "bayesian":
        # select best src as indep
        y_train = np.array([tb[x][A] for x in IC])
        for src in attrs_excA:
            tmp = [tb[x][src] for x in IC]
            X_train.append(np.vander(tmp, params["n_order"] + 1, increasing=True))
            x_train.append(tmp)
        return X_train, y_train, x_train

def init(func_name, y_train, params=None):
    if func_name == "bayesian":
        if params and 'tol' in params and 'init' in params:
            reg = BayesianRidge(tol=params['tol'], fit_intercept=params['fit_intercept'], compute_score=params['compute_score'])
            reg.set_params(alpha_init=params['init'][0], lambda_init=params['init'][1])
        else:
            init = [1 / np.var(y_train), 1.]
            reg = BayesianRidge(tol=random.random()*1e-5, fit_intercept=False, compute_score=True)
            reg.set_params(alpha_init=init[0], lambda_init=init[1])
        return reg
    elif func_name == "linear":
        return LinearRegression()

