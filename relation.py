#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2025/10/24 9:19
@Author  : Tengdi Zheng
@File    : relation.py
"""

from util import *
import warnings
warnings.filterwarnings("ignore")

def Training_model(model, X_train, y_train, threshold):
    """
    A framework of model.

    Parameter
    ----------------
    model : selected model.

    X_train : ndarray of shape (n_train_samples, n_features + 1 (dataset label))
        Data.

    y_train : ndarray of shape (n_train_samples, 2 (survival time, event indicator))
        Target.

    threshold : a const
        A compressibility factor.

    Returns
    ---------------
    :returns: ndarray of (1, n_features)
        Coefficient estimates under the corresponding model.
    """
    coef, intercept = eval(model)(X_train, y_train, threshold)
    W = np.hstack((coef, intercept))
    return W

def w_mse(y_true, y_pred, event_observed):
    """
    Calculate the weighted prediction loss.

    Parameter
    ----------------
    y_true : ndarray of shape (n_test_samples, 1)
        Original survival time.

    y_pred : ndarray of shape (n_test_samples, 1)
        Predict survival time.

    event_observed : ndarray of shape (n_test_samples, 1)
        The event indicator.

    Returns
    ---------------
    :returns: a const
        The weighted prediction loss.
    """
    KMw = Kaplan_Meier_weight(event_observed)
    KMw = KMw * (y_true.shape[0] / np.sum(KMw))
    squared_errors = (y_true - y_pred) ** 2
    w_mse = np.mean(KMw * squared_errors)
    return w_mse

def Relationship_matrix(data, alpha=0.5, theta=0.2, threshold=0.001):
    """
    Calculate the relationship matrix between the datasets.

    Parameter
    ----------------
    data : ndarray of shape (n_samples, n_features + 3 (dataset label, survival time, event indicator))
        Data.

    alpha : a const
        Scaling parameter.

    theta : a const
        Small threshold to truncate small relationship.

    threshold : a const
        Small threshold to truncate small coefficient estimate.

    Returns
    ---------------
    :returns:  ndarray of shape (n_datasets, n_datasets)
        Inter-dataset relationship.
    """
    loss = {}
    for label, data_group in data.groupby(by='dataset'):
        loss[label] = []
        data_group = data_group.reset_index(drop=True)
        train_set, _ = split_normalize_data(data_group, test_size=0.25, random_state=42)
        X_train, y_train = train_set[:, 1:-2].astype(np.float64), train_set[:, -2:].astype(np.float64),
        W = Training_model('AL', X_train, y_train, threshold)
        for label2, data_group2 in data.groupby(by='dataset'):
            data_group2 = data_group2.reset_index(drop=True)
            _, test_set = split_normalize_data(data_group2, test_size=0.25, random_state=42)
            X_test, y_test = test_set[:, 1:-2].astype(np.float64), test_set[:, -2:].astype(np.float64)
            log_y_predict = add_intercept(X_test) @ W
            loss[label].append(w_mse(np.log(y_test[:, 1]), log_y_predict, y_test[:, 0]))

    loss_matrix, rel_matrix = np.identity(len(loss)), np.identity(len(loss))
    row = 0
    for label, data_group in data.groupby(by='dataset'):
        loss_matrix[row, :] = np.array(loss[label])
        row += 1

    for i in range(len(loss)):
        for j in range(len(loss)):
            if i != j:
                rel_matrix[i, j] = e(((loss_matrix[j, i] - loss_matrix[i, i]) / loss_matrix[i, i]) +
                                     ((loss_matrix[i, j] - loss_matrix[j, j]) / loss_matrix[j, j]), alpha)
    np.fill_diagonal(rel_matrix, 0)
    rel_matrix = rel_matrix / (rel_matrix.max() * (1 + theta))
    rel_matrix = np.where(rel_matrix < theta, 0, rel_matrix)
    rel_matrix = pd.DataFrame(rel_matrix, columns=np.unique(data[['dataset']]),
                 index=np.unique(data[['dataset']]))
    return rel_matrix