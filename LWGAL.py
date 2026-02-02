#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Created on 2025/9/16 9:58
@Author  : Tengdi Zheng
@File    : LWGAL.py
"""

import itertools
from scipy.linalg import block_diag
import warnings
import numpy as np
import pandas as pd
from util import Kaplan_Meier_weight

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
    weight = np.ones_like(y[:, 0])
    cancer_name = np.unique(X[:, 0])
    for name in cancer_name:
        index = X[:, 0] == name
        delta = y[index, 0]
        KMw = Kaplan_Meier_weight(delta)
        KMw = KMw * (X.shape[0] / np.sum(KMw))
        weight[index] = KMw
    return weight

class LWGAL():
    def __init__(
            self,
            lamd=1e-3,
            tol=1e-3,
            max_iter=1000,
            verbose=False,
            random_state=42,
            threshold=False
    ):
        '''
        The proposed locally weighted group AFT-lasso (LWGAL) method.
        :param lamd: regularization parameter lambda
        :param tol: early stopping tol
        :param max_iter: maximum iterations s_max
        :param verbose: output result or not
        :param threshold: compressibility factor
        '''
        self.lamd = lamd
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state
        self.threshold = threshold

    def Diagonal_matrix_construction(self, X, y, Aerfa, datasets):
        """
        Data conversion.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features + 1 (dataset label))
            Data.

        y : ndarray of shape (n_samples, 2 (survival time, event indicator))
            Target.

        Aerfa : ndarray of shape (n_datasets, n_datasets)
            weight matrix.

        datasets : ndarray of shape (n_dataset, 1)
            the sample size of each dataset.

        Returns
        -------------
        :returns: the transformed data X, X_b, y
        """

        X_total, X_b_total, y_total = np.array([0]), np.array([0]), np.array([0]).reshape(-1, 1)
        KMw = Muti_weight(X, y)  # obtain the individual weights corresponding to each dataset
        dense_sample_size = []
        for loc in range(len(datasets)):
            aerfa = Aerfa.iloc[:, loc]
            w = []
            for i in range(len(X)):
                for j in range(len(aerfa)):
                    if X[i][0] == aerfa.index[j]:
                        w.append(aerfa[aerfa.index[j]])
            aer = np.matrix(w).reshape(-1, 1)
            aer = np.multiply(aer, KMw.reshape(-1,1))   # two types of weighting: AFT weighting and correlation coefficient matrix weighting
            # aer = aer * (X.shape[0] / np.sum(aer))  # a weighted operation corresponding to the sklearn library
            aer = np.power(aer, 0.5)
            XX, XX_b, yy = np.matrix(X[:, 1:]).astype(np.float64), np.matrix(np.ones(X[:, 1:].shape[0])).reshape(-1, 1), np.log(np.matrix(y[:, 1])).reshape(-1, 1)
            XX, XX_b, yy = np.multiply(aer, XX), np.multiply(aer, XX_b), np.multiply(aer, yy)
            XX, XX_b, yy = XX[np.array(np.sum(aer, axis=1) != 0).reshape(-1, )], XX_b[
                np.array(np.sum(aer, axis=1) != 0).reshape(-1, )], \
                yy[np.array(np.sum(aer, axis=1) != 0).reshape(-1, )]  # remove the samples whose weights sum up to 0, thereby reducing the calculation time.
            dense_sample_size.append(len(XX_b))
            X_total, X_b_total, y_total = block_diag(X_total, XX), block_diag(X_b_total, XX_b), np.concatenate((y_total, yy), axis=0)
        X_total, X_b_total, y_total = X_total[1:, 1:], X_b_total[1:, 1:], y_total[1:, :]
        return X_total, X_b_total, y_total, dense_sample_size

    def Group_identification(self, datasets, p):
        """
        Group_identification.

        Parameters
        ----------------
        datasets : ndarray of shape (n_dataset, 1)
            The sample size of each dataset.

        p : a const
            Number of covariates.

        Returns
        -------------
        :returns:  ndarray of shape (p * M,)
            Group identification.
        """
        group_id = []
        for i in range(len(datasets)):
            for j in range(p):
                group_id.append(j)  # the labels of a group are the same
        return group_id

    def model(self, X, X_b, y, dense_sample_size, group_id, p, M):
        """
        Fit model with block coordinate descent.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features + 1 (dataset label))
            Data.

        X_b : ndarray of (n_samples, )
            Data_0.

        y : ndarray of shape (n_samples, 2 (survival time, event indicator))
            Target.

        dense_sample_size : a list
            The remaining effective sample size of each dataset.

        group_id : ndarray of shape (p * M,)
            group identification.

        p : Number of covariates.

        M : Number of datasets.

        Returns
        -------
        :returns: object
            Fitted estimator.
        """
        assert X.shape[1] == p * M, \
            'The row number of `X` must be the same as `p * M` '
        w, b = np.matrix(np.zeros((M * p, 1))), np.matrix(np.zeros(M)).reshape(-1, 1)
        r = y
        r_old, norm_old = r, np.Inf
        niter = itertools.count(1)
        Assit = [[] for row in range(M)]
        P_col = [[] for row in range(M)]
        pp = X.shape[1]
        for loc in range(M):
            X_l = X[int(np.sum(dense_sample_size[:loc])): int(np.sum(dense_sample_size[:loc+1])), loc * (pp//M): (loc+1) * (pp//M)]
            for cov in range(pp//M):
                if X_l[:, cov].T @ X_l[:, cov] == 0:
                    Assit[loc].append(np.zeros(len(X_l)).reshape(1, -1))
                else:
                    Assit[loc].append((1/(X_l[:, cov].T @ X_l[:, cov]) * X_l[:, cov].T).reshape(1,-1))
                P_col[loc].append(X_l[:, cov].reshape(-1,1) @ Assit[loc][cov])
        #####################################Start the iteration process###################################
        for it in niter:
            norm = 0
            for cov in range(p):
                cov_group = [x for x, y in list(enumerate(group_id)) if y == cov]
                r_cov = r + X[:, cov_group] * w[cov_group]

                # #####################Calculate the orthogonal projection matrix##########################
                # XX_zj = X[:, cov_group] @ np.linalg.pinv(X[:, cov_group].T @ X[:, cov_group]) @ X[:, cov_group].T
                # XiTri = np.sqrt(sum(e ** 2 for e in XX_zj @ r_cov))
                # if XiTri < self.lamd / 2:
                #     w[cov_group] = np.matrix(np.zeros(M)).reshape(-1, 1)
                # else:
                #     w[cov_group] = (1 - self.lamd / (2 * XiTri))[0,0] * (
                #             np.linalg.pinv(X[:, cov_group].T @ X[:, cov_group]) * (X[:, cov_group].T * r_cov))

                ########################Block computing##########################################
                Pri, XiTri = 0, np.matrix(np.zeros(M)).reshape(-1, 1)
                for i in range(M):
                    Pri += np.linalg.norm(P_col[i][cov] @ r_cov[int(np.sum(dense_sample_size[:i])): int(np.sum(dense_sample_size[:i+1]))], ord=2)**2
                    XiTri[i, :] = Assit[i][cov] * r_cov[int(np.sum(dense_sample_size[:i])): int(np.sum(dense_sample_size[:i+1]))]
                Pri = np.sqrt(Pri)
                if Pri < self.lamd / 2:
                    w[cov_group] = np.matrix(np.zeros(M)).reshape(-1, 1)
                else:
                    w[cov_group] = (1 - self.lamd / (2 * Pri)) * XiTri

                r = r_cov - X[:, cov_group] * w[cov_group]
                norm += np.linalg.norm(X[:, cov_group] * w[cov_group], ord=2)

            r_b = r + X_b * b
            b = np.linalg.pinv(X_b.T @ X_b) * (X_b.T @ (y - X @ w))
            r = r_b - X_b * b
            delta_old = np.linalg.norm(r_old, ord=2) ** 2 + self.lamd * norm_old
            delta = np.linalg.norm(r, ord=2) ** 2 + self.lamd * norm
            if abs(delta_old - delta) / delta_old < self.tol or it > self.max_iter:
                break
            else:
                r_old, norm_old = r, norm
                if self.verbose:
                    if it % 10 == 0:
                        print('Iteration: {}, delta = {}'.format(it, delta))
        coef = np.array(w).reshape(M, p)

        if self.threshold:
            coef_log = abs(coef) < self.threshold
            coef[:, pd.DataFrame(coef_log).eq(True).all().values.tolist()] = 0

        self.coef_ = coef.reshape(-1, 1)
        self.intercept_ = b
        return self

    def fit(self, X, y, Aerfa, datasets, p, M):
        """
        training model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features + 1 (dataset label))
            Data.

        y : ndarray of shape (n_samples, 2 (survival time, event indicator))
            Target.

        Aerfa : ndarray of shape (n_datasets, n_datasets)
            weight matrix.

        datasets : ndarray of shape (n_dataset, 1)
            The sample size of each dataset.

        p : a const
            Number of variables.

        M : a const
            Number of datasets.

        Returns
        -------
        :returns: object
            Fitted estimator.

        Notes
        -----
        Block coordinate descent is an algorithm that only one axis direction is
        optimized in each iteration and the values of other axes are fixed,
        so that the multi-variable optimization problem becomes a univariate
        optimization problem.

        In order to improve the calculation speed, some matrices that need to
         be reused are calculated in advance.

         For details, please refer to the paper Simon N , Tibshirani R .
         Standardization and the Group Lasso Penalty[J].Statistica Sinica, 2012,
          22(3):983-1001.
        """

        if self.lamd == 0 and sum(sum(np.triu(Aerfa, 1))):
            warnings.warn(
                "With lamda=0 and h<np.min(distance), You are advised to use "
                "the LinearRegression for each dataset",
                stacklevel=2
            )

        elif self.lamd == 0:
            warnings.warn(
                "With lamda=0, You are advised to use local weighting "
                "algorithm",
                stacklevel=2,
            )

        elif sum(sum(np.triu(Aerfa, 1))) == 0:
            warnings.warn(
                "With h<np.min(distance), You are advised to use Lasso for "
                "each dataset",
                stacklevel=2,
            )

        if isinstance(X, pd.DataFrame):
            X = X.values

        if isinstance(y, pd.Series):
            y = y.values

        # make changes to the data
        X, X_b, y, dense_sample_size = \
            self.Diagonal_matrix_construction(X, y, Aerfa, datasets)

        # the dataset identification of the corresponding variable group
        group_id = \
            self.Group_identification(datasets, p)

        return self.model(X, X_b, y, dense_sample_size, group_id, p, M)

    def predict(self, X, loc, p):
        """
        Prediction

        Parameter
        ----------------
        X : ndarray of shape (n_samples, n_features + 1 (dataset label))
            Data.

        loc : a const
            Dataset label.

        p : a const
            Number of covariates.

        Returns
        ---------------
        :returns: ndarray of (n_m_sample,)
            Predict result.
        """
        W = np.vstack((self.coef_[loc*p: (loc+1)*p, :], self.intercept_[loc]))
        return np.exp(np.array(X @ W)).reshape(-1, 1)