#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Created on 2025/11/5 9:43
@Author  : Tengdi Zheng
@File    : generated_data.py
"""

import numpy as np
import pandas as pd
import random
import warnings

def get_G_cov_AR(dim):
    """
    Generate G variables with auto-regressive-related structure.

    Parameters
    ----------
    dim : a constant
        The number of G covariates.

    Returns
    -------
    :returns:  ndarray of shape (n_features, n_features)
            Covariance matrix with auto-regressive-related structure.
    """
    cov = np.zeros([dim,dim])
    for i in range(dim):
        for j in range(dim):
                cov[i,j] = 0.3**abs(i-j)
    return cov

def get_C_cov_AR(dim):
    """
    Generate C variables with auto-regressive-related structure.

    Parameters
    ----------
    dim : a constant
        The number of C covariates.

    Returns
    -------
    :returns:  ndarray of shape (n_features, n_features)
            Covariance matrix with auto-regressive-related structure.
    """
    cov = np.zeros([dim,dim])
    for i in range(dim):
        for j in range(dim):
                cov[i,j] = 0.2**abs(i-j)
    return cov

def get_G_cov_Band(dim):
    """
    Generate G variables with banded-related structure.

    Parameters
    ----------
    dim : a constant
        The number of G covariates.

    Returns
    -------
    :returns:  ndarray of shape (n_features, n_features)
            Covariance matrix with banded-related structure.
    """
    cov = np.zeros([dim,dim])
    for i in range(dim):
        for j in range(dim):
            if np.abs(i - j) == 0:
                cov[i, j] = 1
            elif np.abs(i - j) == 1:
                cov[i, j] = 0.33
            elif np.abs(i - j) == 2:
                cov[i, j] = 0.15
    return cov

def get_C_cov_Band(dim):
    """
    Generate C variables with banded-related structure.

    Parameters
    ----------
    dim : a constant
        The number of C covariates.

    Returns
    -------
    :returns:  ndarray of shape (n_features, n_features)
            Covariance matrix with banded-related structure.
    """
    cov = np.zeros([dim,dim])
    for i in range(dim):
        for j in range(dim):
            if np.abs(i - j) == 0:
                cov[i, j] = 1
            elif np.abs(i - j) == 1:
                cov[i, j] = 0.15
    return cov

def get_mean(dim):
    """
    Generate mean vector.

    Parameters
    ----------
    dim : a constant
        The number of covariates.

    Returns
    -------
    :returns:  ndarray of shape (1, n_features)
        Mean vector.
    """
    return np.zeros(dim)

def get_one_data(n,pc,pg,pindex,low_signal, high_signal, correlation_structure):
    """
    Generate a dataset.

    Parameters
    ----------
    n : a constant
        The average sample size in each dataset.

    pc : a const
        The number of C covariates.

    pg : a const
        The number of G covariates.

    pindex : a list
        Indicate the location of the significant G covariates.

    low_signal : a const
        The lower bound of uniform distribution.

    high_signal : a const
        The upper bound of uniform distribution.

    correlation_structure : a string
        Indicate the correlation structure of covariates

    Returns
    -------
    :returns:  ndarray of shape (1, n_features)
        Mean vector.
    """
    if correlation_structure == 'auto-regressive':
        Cvar = np.random.multivariate_normal(get_mean(pc), get_C_cov_AR(pc), n, 'raise')  # generate multivariate normal outpatient variables
        Gvar = np.random.multivariate_normal(get_mean(pg), get_G_cov_AR(pg), n, 'raise')  # generate multivariate normal genetic data
    elif correlation_structure == 'banded':
        Cvar = np.random.multivariate_normal(get_mean(pc), get_C_cov_Band(pc), n, 'raise')  # generate multivariate normal outpatient variables
        Gvar = np.random.multivariate_normal(get_mean(pg), get_G_cov_Band(pg), n, 'raise')  # generate multivariate normal genetic data
    else:
        warnings.warn(
            "Please using the correct correlation_structure: ['auto-regressive', 'banded']"
            "correlation_structure",
            stacklevel=2,
        )
    bias = np.ones([n, 1])
    X = np.concatenate((Cvar, Gvar, bias), axis=1)
    index = np.zeros(pc + pg + 1)
    index[:pc] = 1
    index[pc+pg] = 1
    index[pindex] = 1
    beta = np.zeros(pc + pg + 1)
    for i in range(pc+pg+1):
        if index[i] == 1:
            beta[i] = random.uniform(low_signal, high_signal)

    log_duration = X.dot(beta) + np.random.normal(0, 0.1, size=X.shape[0])  # generate survival data using the logarithmic AFT model
    log_censoring_times = np.random.uniform(0, np.max(log_duration), size=X.shape[0])
    duration, censoring_times = np.exp(log_duration), np.exp(log_censoring_times)

    observed_duration = np.minimum(duration, censoring_times)
    event_observed = (observed_duration == duration).astype(int)
    Y = np.vstack((event_observed, duration)).T
    return X, Y, beta


def simulation_data(n, ns, wm, sm, pc, pg, indp, low_signal_a, low_signal_b, high_signal_a, high_signal_b, relationship_structure, correlation_structure):
    """
    Generate simulated data.

    Parameters
    ----------
    n : a constant
        The average sample size in each dataset.

    ns : a constant
        The standard deviation of the average sample size in each dataset.

    wm : a const
        The number of weak signal datasets.

    sm : a const
        The number of strong signal datasets.

    pc : a const
        The number of C covariates.

    pg : a const
        The number of G covariates.

    indp : a const
        The number of important G covariates.

    low_signal_a : a const
        The lower bound of uniform distribution of weak signal level.

    low_signal_a : a const
        The upper bound of uniform distribution of weak signal level.

    high_signal_a : a const
        The lower bound of uniform distribution of strong signal level.

    high_signal_a : a const
        The upper bound of uniform distribution of strong signal level.

    relationship_structure : a string
        Indicate the relationship structure across datasets.

    correlation_structure : a string
        Indicate the correlation structure of covariates.

    Returns
    -------
    :returns:  the entire simulated data
    """
    simdat, simdat1, simdat2 = [], [], []
    beta, bet1, bet2 = [], [], []
    N = np.random.normal(n, ns, wm + sm).astype(int)
    l = wm + sm
    pindex = random.sample(range(pc, pc + pg), indp)
    if relationship_structure == 'R1':
        for loc in range(wm):
            x1, y1, beta1 = get_one_data(N[loc], pc, pg, pindex, (1 + 2 * loc / l) * low_signal_a,
                                         (1 + 2 * loc / l) * low_signal_b, correlation_structure)
            data = np.hstack((x1[:, :-1], y1))
            simdat.append(data)
            beta.append(beta1)

        for loc in range(sm):
            x2, y2, beta2 = get_one_data(N[loc], pc, pg, pindex, (1 + 2 * loc / l) * high_signal_a,
                                         (1 + 2 * loc / l) * high_signal_b, correlation_structure)
            data = np.hstack((x2[:, :-1], y2))
            simdat.append(data)
            beta.append(beta2)
    elif relationship_structure == 'R2':
        for loc in range(wm):
            x1, y1, beta1 = get_one_data(N[loc], pc, pg, pindex,
                                         (1 - 2 * loc / l) * low_signal_a + (2 * loc / l) * high_signal_a,
                                         (1 - 2 * loc / l) * low_signal_b + (2 * loc / l) * high_signal_b, correlation_structure)
            data1 = np.hstack((x1[:, :-1], y1))
            simdat1.append(data1)
            bet1.append(beta1)

        for loc in range(sm):
            x2, y2, beta2 = get_one_data(N[loc], pc, pg, pindex,
                                         (1 - 2 * loc / l) * low_signal_a + (2 * loc / l) * high_signal_a,
                                         (1 - 2 * loc / l) * low_signal_b + (2 * loc / l) * high_signal_b, correlation_structure)
            data2 = np.hstack((x2[:, :-1], y2))
            simdat2.insert(0, data2)
            bet2.insert(0, beta2)
        simdat = simdat1 + simdat2
        beta = bet1 + bet2
    else:
        warnings.warn(
            "Please using the correct relationship structure: ['R1', 'R2']"
            "relationship_structure",
            stacklevel=2,
        )


    threshold = np.unique(beta)[1] if len(np.unique(beta)) > 1 else np.unique(beta)[0]
    threshold = threshold / 3

    data_total = pd.DataFrame(simdat[0])
    data_total.insert(0, 'dataset', 'dataset101')

    for i in range(1, wm+sm):
        data = pd.DataFrame(simdat[i])
        data.insert(0, 'dataset', 'dataset'+str(i+101))
        data_total = pd.concat([data_total, data])
    data_total.rename(columns={pc+pg: 'indicate'}, inplace=True)
    data_total.rename(columns={pc+pg+1: 'y'}, inplace=True)
    data_total = data_total.sort_values(by='y', ascending=True)
    return data_total, np.array(beta), threshold, N