#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2025/9/15 10:07
@Author  : Tengdi Zheng
@File    : util.py
"""

import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
import deepsurv
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
import math

def add_intercept(X):
    """
    Add intercept to X.

    Parameters
    ----------
     X : ndarray of (n_samples, n_features)
            Data.

    Returns
    -------
    :returns:  ndarray of (n_samples, n_features + 1)
            [X, X_0]
    """
    return np.c_[X, np.ones(len(X))]

def Kernel(distance_matrix, gwr_bw, dataset, kernel):
    """
    Kernel_function.

    Parameters
    ----------
    distance_matrix : ndarray of shape (n_datasets, n_datasets)
        The distance between datasets.

    gwr_bw : a constant
        Bandwidth.

    dataset : a string list
        The name of datasets.

    kernel : a string
        Specify different kernel functions ('threshold', 'bi-square', 'gaussian', 'exponential').

    Returns
    -------
    :returns:  ndarray of shape (n_datasets, n_datasets)
            Dataset-level weight matrix.
    """
    aerfa = np.ones_like(distance_matrix)
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[0]):
            if i == j:
                continue
            elif distance_matrix[i][j] > gwr_bw:
                aerfa[i][j] = 0
            else:
                if kernel == 'threshold': aerfa[i][j] = distance_matrix[i][j]
                if kernel == 'bi-square': aerfa[i][j] = pow(1 - pow((distance_matrix[i][j] / gwr_bw), 2), 2)
                if kernel == 'gaussian': aerfa[i][j] = np.exp(-0.5 * pow((distance_matrix[i][j] / gwr_bw), 2))
                if kernel == 'exponential': aerfa[i][j] = np.exp(-distance_matrix[i][j] / gwr_bw)
    return pd.DataFrame(aerfa, columns=dataset,
                 index=dataset)

def Kaplan_Meier_weight(OS):
    """
    Calculate the KMw weights for each dataset.

    Parameter
    ----------------
    OS : ndarray of shape (n_m_samples, 2 (survival time, event indicator))
        Survival status.

    Returns
    ---------------
    :returns: ndarray of (n_m_samples, n_m_samples)
        KM Weighted matrix for each dataset.
    """
    n, delta, KMw, cp = OS.shape[0], np.array(OS), np.ones(OS.shape[0]), np.ones(OS.shape[0])
    KMw[0], cp[0] = delta[0] / n, ((n-1)/n)**delta[0]
    for i in range(1, n): KMw[i], cp[i] = delta[i] * cp[i-1] / (n - i), cp[i-1] * ((n-i-1)/(n-i))**delta[i]
    return KMw

def Muti_weight(X, y):
    """
    Obtain the sample weights of different datasets.

    Parameter
    ----------------
    X : ndarray of shape (n_samples, n_features + 1 (dataset label))
        Data.

    y : ndarray of shape (n_samples, 2 (survival time, event indicator))
        Target.

    Returns
    ---------------
    :returns: ndarray of (n_samples, n_samples)
        KM Weighted matrix.
    """
    weight = np.ones_like(y[:,0])
    cancer_name = np.unique(X[:, 0])
    for name in cancer_name:
        index = X[:, 0] == name
        delta = y[index, 0]
        KMw = Kaplan_Meier_weight(delta)
        KMw = KMw * (X.shape[0] / np.sum(KMw))
        weight[index] = KMw
    return weight

def AR(X, y):
    """
    (1) The implementation of the AFT-Ridge model.

    Parameter
    ----------------
    X : ndarray of shape (n_m_samples, n_features + 1 (dataset label))
        Data.

    y : ndarray of shape (n_m_samples, 2 (survival time, event indicator))
        Target.

    Returns
    ---------------
    :returns: ndarray of (1, n_features + 1 (intercept))
        The coefficient estimation results of AR.
    """
    X_Ridge = X.astype(np.float64)
    y_Ridge = np.log(y[:, 1])
    delta = y[:, 0]
    KMw = Kaplan_Meier_weight(delta)
    KMw = KMw * (X.shape[0] / np.sum(KMw))
    index = np.where(KMw != 0)
    X_Ridge, y_Ridge, KMw = X_Ridge[index], y_Ridge[index], KMw[index]
    alphas_range = np.logspace(-4, 1, 200, base=10)
    estimator = RidgeCV(alphas=alphas_range, cv=5, fit_intercept=True)
    beta = estimator.fit(X_Ridge, y_Ridge, sample_weight=KMw)
    return beta.coef_, beta.intercept_

def AL(X, y, threshold):
    """
    (2) The implementation of the AFT-Lasso model and the AFT-glasso model.

    Parameter
    ----------------
    X : ndarray of shape (n_m_samples, n_features + 1 (dataset label))
        Data.

    y : ndarray of shape (n_m_samples, 2 (survival time, event indicator))
        Target.
        
    threshold: a const
        A compressibility factor.

    Returns
    ---------------
    :returns: ndarray of (1, n_features + 1 (intercept))
        The coefficient estimation results of AL or gAL.
    """
    X_Lasso = X.astype(np.float64)
    y_Lasso = np.log(y[:, 1])
    delta = y[:, 0]
    KMw = Kaplan_Meier_weight(delta)
    KMw = KMw * (X.shape[0] / np.sum(KMw))
    index = np.where(KMw != 0)
    X_Lasso, y_Lasso, KMw = X_Lasso[index], y_Lasso[index], KMw[index]
    alphas_range = np.logspace(-4, 1, 200, base=10)
    estimator = LassoCV(alphas=alphas_range, cv=5, max_iter=1000, fit_intercept=True)
    beta = estimator.fit(X_Lasso, y_Lasso, sample_weight=KMw)
    beta.coef_[abs(beta.coef_) < threshold] = 0
    return beta.coef_, beta.intercept_


def GWAL(X, y, aerfa, threshold):
    """
    (3) The implementation of the l1-penalized GWAFT model (Cai et al., 2024).

    Parameter
    ----------------
    X : ndarray of shape (n_m_samples, n_features + 1 (dataset label))
        Data.

    y : ndarray of shape (n_m_samples, 2 (survival time, event indicator))
        Target.
    
    aerfa : ndarray of shape (n_datasets, n_datasets)
        Dataset-level weight matrix.

    threshold : a const
        A compressibility factor.

    Returns
    ---------------
    :returns: ndarray of (1, n_features + 1 (intercept))
        The coefficient estimation results of GWAL.
    """
    KMw = Muti_weight(X, y)
    X, wt = X, []
    for i in range(len(X)):
        for j in range(len(aerfa)):
            if X[i][0] == aerfa.index[j]:
                wt.append(aerfa[aerfa.index[j]])
    X_Lasso = X[:, 1:].astype(np.float64)
    y_Lasso = np.log(y[:, 1])
    w = np.array(wt * KMw)
    index = np.where(w != 0)
    X_Lasso, y_Lasso, w = X_Lasso[index], y_Lasso[index], w[index]
    alphas_range = np.logspace(-4, 1, 200, base=10)
    estimator = LassoCV(alphas=alphas_range, cv=5, max_iter=1000, fit_intercept=True)
    beta = estimator.fit(X_Lasso, y_Lasso, sample_weight=w)
    beta.coef_[abs(beta.coef_) < threshold] = 0
    return beta.coef_, beta.intercept_

def RSF(X, y):
    """
    (5) The implementation of the Random Survival Forest method.

    Parameter
    ----------------
    X : ndarray of shape (n_m_samples, n_features + 1 (dataset label))
        Data.

    y : ndarray of shape (n_m_samples, 2 (survival time, event indicator))
        Target.

    Returns
    ---------------
    :returns: the fitted RSF model.
    """
    rsf = RandomSurvivalForest(
        n_estimators=1000, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=42
    )
    # List of tuples
    aux = [(e1, e2) for e1, e2 in y]
    # Structured array
    y = np.array(aux, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    rsf.fit(X, y)
    return rsf


def DS(X, y):
    """
    (6) The implementation of the Deep survival method method.

    Parameter
    ----------------
    X : ndarray of shape (n_m_samples, n_features + 1 (dataset label))
        Data.

    y : ndarray of shape (n_m_samples, 2 (survival time, event indicator))
        Target.

    Returns
    ---------------
    :returns: the fitted DS model.
    """
    train_data = {'x': X.astype('float32'), 't': y[:, 1].astype('float32'), 'e': y[:, 0].astype('int32')}
    network = deepsurv.DeepSurv(n_in=X.shape[1], learning_rate=1e-3)
    network.train(train_data=train_data, n_epochs=500, verbose=False)
    return network

def normlized_data(data):
    """
    Data standardization.

    Parameter
    ----------------
    data : ndarray of shape (n_m_samples, n_features + 3 (dataset label, survival time, event indicator))
        Data.

    Returns
    ---------------
    :returns: Standardized data in each dataset.
    """
    transfer = StandardScaler()
    data[:, 1:-2] = transfer.fit_transform(data[:, 1:-2])
    return data

def spilt_data(data, test_size, random_state):
    """
    Data spilt.

    Parameter
    ----------------
    data : ndarray of shape (n_m_samples, n_features + 3 (dataset label, survival time, event indicator))
        Data.

    test_size : a const
        The size of test set.

    random_state : a const
        Random seed.

    Returns
    ---------------
    :returns: Data division based on survival or death.
    """
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in split.split(data, data[:, -2]):
        train_set = data[train_index]
        test_set = data[test_index]  # 保证测试集
    return normlized_data(train_set), normlized_data(test_set)

def split_normalize_data(data, test_size, random_state):
    """
    Standardize each datasets separately.

    Parameter
    ----------------
    data : ndarray of shape (n_samples, n_features + 3 (dataset label, survival time, event indicator))
        Data.

    test_size : a const
        The size of test set.

    random_state : a const
        Random seed.

    Returns
    ---------------
    :returns: Data division based on survival or death.
    """
    subtypes = np.unique(data.iloc[:, 0])
    train, test = [], []
    for name in subtypes:
        data_subtype = data[data.iloc[:, 0] == name].reset_index(drop=True)
        train_set, test_set = spilt_data(np.array(data_subtype), test_size, random_state)
        train.append(train_set), test.append(test_set)
    train, test = np.concatenate(train), np.concatenate(test)
    return train[np.argsort(train[:, -1])], test[np.argsort(test[:, -1])]

def e(x, alpha):
    """
    Relationship calculation.

    Parameter
    ----------------
    X : a const
        Calculation results of predicted loss error.

    alpha : a const
        Scaling parameter.

    Returns
    ---------------
    :returns: ndarray of shape (n_datasets, n_datasets)
            Dataset-level relationship matrix.
    """
    return math.exp(-alpha * x)

def save_files(data, save_name):
    """
    Save the necessary files.

    Parameter
    ----------------
    data : data

    save_name : file name

    Returns
    ---------------
    :returns:  Saved files.
    """
    pd.DataFrame(data).to_excel(save_name+'.xlsx')

def check_hyperparameter(sel_lambda, sel_bw, lambda_total, bw_total):
    """
    Check whether the selected hyperparameters are at the boundary points.

    Parameter
    ----------------
    sel_lambda : selected lambda

    sel_bw : selected bw

    lambda_total : lambda list

    bw_total : bw list

    Returns
    ---------------
    :returns:  True or False.
    """
    if sel_lambda == lambda_total[0] or sel_lambda == lambda_total[-1]:
        return True
    else:
        return False