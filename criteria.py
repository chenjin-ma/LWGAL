#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Created on 2025/9/6 10:07
@Author  : Tengdi Zheng
@File    : criteria.py
"""

import numpy as np

class Criteria():
    def __init__(
            self,
            MAB = None,
            MSE = None,
            TP = None,
            Total = None,
            Tpr = None,
            Fpr = None
    ):
        """
        Evaluation criteria calculation.
        :param MAB: mean absolute error
        :param MSE: mean squared error
        :param TP: true positives
        :param Total: model size
        :param Tpr: true positive rate
        :param Fpr: false positive rate
        """
        self.MAB = MAB
        self.MSE = MSE
        self.TP = TP
        self.Total = Total
        self.Tpr = Tpr
        self.Fpr = Fpr

    def MAB_process(self, estimator, intercept, true):
        """
        The calculation of MAB.

        Parameters
        ----------
        estimator : coefficient estimates

        intercept : the intercept estimates

        true : coefficient truth value

        Returns
        -------
        :returns:  the MAB loss
        """
        estimator = np.hstack((estimator, intercept.reshape(-1,1)))
        return np.mean(np.abs(estimator - true))

    def MSE_process(self, estimator, intercept, true):
        """
        The calculation of MSE.

        Parameters
        ----------
        estimator : coefficient estimates

        intercept : the intercept estimates

        true : coefficient truth value

        Returns
        -------
        :returns:  the MSE loss
        """
        estimator = np.hstack((estimator, intercept.reshape(-1, 1)))
        return np.mean((estimator - true) ** 2)

    def TP_process(self, estimator, beta):
        """
        The calculation of TP.

        Parameters
        ----------
        estimator : coefficient estimates

        beta : coefficient truth value

        Returns
        -------
        :returns:  the TP result
        """
        return ((beta[:,:-1] != 0) & (estimator != 0)).sum()

    def TPR_process(self, estimator, beta):
        """
        The calculation of TPR.

        Parameters
        ----------
        estimator : coefficient estimates

        beta : coefficient truth value

        Returns
        -------
        :returns:  the TPR result
        """
        return self.TP_process(estimator,beta) / (beta[:, :-1] != 0).sum()

    def Total_process(self, estimator):
        """
        The calculation of MS.

        Parameters
        ----------
        estimator : coefficient estimates

        Returns
        -------
        :returns:  the MS result
        """
        return (estimator != 0).sum()

    def FPR_process(self, estimator, beta):
        """
        The calculation of FPR.

        Parameters
        ----------
        estimator : coefficient estimates

        beta : coefficient truth value

        Returns
        -------
        :returns:  the FPR result
        """
        return (self.Total_process(estimator)-self.TP_process(estimator,beta))/(beta[:,:-1] == 0).sum()

    def estimation_process(self, estimator, intercept, true):
        """
        Evaluation of coefficient estimation results.

        Parameters
        ----------
        estimator : coefficient estimates

        intercept : the intercept estimates

        true : coefficient truth value

        Returns
        -------
        :returns:  [MAE, MSE]
        """
        mab = self.MAB_process(estimator, intercept, true)
        mse = self.MSE_process(estimator, intercept, true)
        return [mab, mse]

    def selection_process(self, estimator, beta):
        """
        Evaluation of variable selection results.

        Parameters
        ----------
        estimator : coefficient estimates

        beta : coefficient truth value

        Returns
        -------
        :returns:  [TP, MS, TPR, FPR]
        """
        tp = self.TP_process(estimator, beta)
        total = self.Total_process(estimator)
        tpr = self.TPR_process(estimator,beta)
        fpr = self.FPR_process(estimator,beta)
        return [tp, total, tpr, fpr]



