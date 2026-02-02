#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Created on 2025/12/14 13:07
@Author  : Tengdi Zheng
@File    : sim.py
"""

from LWGAL import *
from util import *
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index
from sksurv.linear_model import CoxPHSurvivalAnalysis # (5) the implementation of the Cox-ridge model
from generated_data import simulation_data  # generate simulated data
from criteria import Criteria  # evaluation criteria
from relation import Relationship_matrix  # correlation matrix across datasets
import warnings
import os
warnings.filterwarnings("ignore")

def parallel_training(hyperparam_index, lamda, bw, distance_matrix, dataset):
    """
    Parallel processes during the selection of hyperparameters.

    Parameters
    ----------
    hyperparam_index : hyperparamter_grid
    
    lamda : regulation parameter
    
    bw : bandwidth
    
    distance_matrix : 1 - r
    
    dataset : the name of datasets.

    Returns
    -------
    :returns:  the cindex list under a set of hyperparameters
    """
    i, j = hyperparam_index[1]['lamda'], hyperparam_index[1]['bw']
    lamd, gwr_bw = lamda[i], bw[j]
    print("lamd=", lamd, "bw=", gwr_bw, ",calcaulating...")
    Aerfa = Kernel(distance_matrix, gwr_bw, dataset, 'bi-square')
    C_index_GWAL, C_index_LWGAL = np.zeros([n_fold, M]), np.zeros([n_fold, M])
    kfolder = KFold(n_splits=n_fold, shuffle=True, random_state=42+random_state)
    fold = 0
    for train_index, test_index in kfolder.split(X_train_o, y_train_o):
        X_train_x, X_test_c = X_train_o[train_index], X_train_o[test_index]
        y_train_x, y_test_c = y_train_o[train_index], y_train_o[test_index]
        LWGAL_model = LWGAL(lamd=lamd, tol=0.001, max_iter=1000, threshold=threshold, verbose=False)
        LWGAL_model.fit(X_train_x, y_train_x, Aerfa, datasets, p, M)
        for loc in range(len(dataset)):
            aerfa = Aerfa.iloc[:, loc]
            XX_test, yy_test = X_test_c[list((X_test_c[:, 0] == dataset[loc])), :][:, 1:].astype(np.float64), \
                               y_test_c[list((X_test_c[:, 0] == dataset[loc]))]
            if len(XX_test) == 0:
                continue

            #################################GWAL###################################
            coef, intercept = GWAL(X_train_x, y_train_x, aerfa, threshold)
            W = np.hstack((coef, intercept))
            y_GWAL = np.exp(add_intercept(XX_test) @ W)
            C_index_GWAL[fold, loc] = concordance_index(yy_test[:, 1], y_GWAL, event_observed=yy_test[:, 0])

            ####################################LWGAL####################################
            y_LWGAL = LWGAL_model.predict(add_intercept(XX_test), loc, p).flatten()
            C_index_LWGAL[fold, loc] = concordance_index(yy_test[:, 1], y_LWGAL, event_observed=yy_test[:, 0])

        fold += 1
    return [i, j, np.average(C_index_GWAL, axis=0) if len(C_index_GWAL) != 0 else 0,
            np.average(C_index_LWGAL, axis=0) if len(C_index_LWGAL) != 0 else 0]


''' Simulation Settings (the default setting is Scenario 1) '''
pc, pg, indp = 5, 495, 20  # number of clinical data (important), gene data [495, 995], and important genes
n, ns = 300, 60  # average sample size for each dataset and SD
wm, sm = 3, 2  # number of weak signal datasets and strong signal datasets [10, 10]
low_signal_a, low_signal_b, high_signal_a, high_signal_b = 0.03, 0.04, 0.11, 0.12  # the bound of the uniform distribution of two signal level
M = wm + sm  # the number of datasets
relationship_structure = 'R2'  # the structure of relationship matrix: ['R1', 'R2']
correlation_structure = 'auto-regressive'  # the structure of correlation: ['auto-regressive', 'banded']
n_fold = 5  # cross_validation folds
repeat_number, effective_repeat = 500, 100  # random partition frequency, effective repetition
''''''''''''''''''''


''' Simulation beginning '''
C_index_Cox, C_index_AR, C_index_AL, C_index_gAL, C_index_GWAL, C_index_RSF, C_index_DS, C_index_LWGAL = \
    np.zeros((effective_repeat, M)), np.zeros((effective_repeat, M)), \
    np.zeros((effective_repeat, M)), np.zeros((effective_repeat, M)), \
    np.zeros((effective_repeat, M)), np.zeros((effective_repeat, M)), \
    np.zeros((effective_repeat, M)), np.zeros((effective_repeat, M))
coef_MAB_loss = np.zeros([6, effective_repeat])
coef_MSE_loss = np.zeros([6, effective_repeat])
TP = np.zeros([4, effective_repeat])
Model_Size = np.zeros([4, effective_repeat])
Cindex = np.zeros([8, effective_repeat])

eff = 0  # record the number of valid repetitions

for random_state in range(repeat_number):
    try:
        if eff >= effective_repeat:  # stop when the desired number of repetitions is met
            break
        print("repeat number:", str(eff+1))

        dataset = ['dataset' + str(i + 100) for i in range(1, wm + sm + 1)]
        data, beta, threshold, N_samples = simulation_data(n, ns, wm, sm, pc, pg, indp, low_signal_a, low_signal_b, high_signal_a, high_signal_b, relationship_structure, correlation_structure)
        relationship_matrix = Relationship_matrix(data, alpha=0.5, theta=0.3, threshold=threshold)
        print('threshold:', threshold, 'censoring rate:', ((data.shape[0] - np.sum(data[['indicate']])) / data.shape[0])[0],
              'N_samples:', N_samples)

        output_dir = 'simulation_data/repeat_' + str(eff + 1)
        if output_dir[-1] != "/":
            output_dir += "/"
        if not os.path.isdir(output_dir):
            print("Directory doesn't exist, creating it")
            os.mkdir(output_dir)
        save_files(data, output_dir + 'sim_data'), save_files(beta, output_dir + 'beta'), save_files(
            relationship_matrix, output_dir + 'rel_matrix')

        p = pc + pg  # the number of covariates
        p1, p2 = p, 0
        datasets = data['dataset'].value_counts()  # the number of datasets and their sample size
        distance_matrix = 1 - np.array(relationship_matrix)
        b_min, b_max = np.min(distance_matrix), np.max(distance_matrix)
        bw = np.linspace(b_min - 0.01, b_max - 0.3, 5)  # give the bandwidth range
        lamda = np.linspace(0.1, 5, 10)  # after the initial screening of np.linspace(0.001, 20, 20)

        cindex_par = np.zeros([2, len(lamda) * len(bw)])
        train_set, test_set = split_normalize_data(data, test_size=0.25, random_state=42+random_state)
        X_train_o, y_train_o = np.array(train_set[:, :-2]), np.array(train_set[:, -2:]).astype('float64')
        X_test_o, y_test_o = np.array(test_set[:, :-2]), np.array(test_set[:, -2:]).astype('float64')

        model_ls = []
        param_range = {'lamda': range(len(lamda)), 'bw': range(len(bw))}
        param_grid = ParameterGrid(param_range)
        with Parallel(n_jobs=1) as parallel:
           model_ls = parallel(delayed(parallel_training)
                           (hyperparam_index, lamda, bw, distance_matrix,
                            dataset) for hyperparam_index in enumerate(param_grid))

        for i, j, GWAL_cindex, LWGAL_cindex in model_ls:
            cindex_par[0, i * len(bw) + j],cindex_par[1, i * len(bw) + j] = \
                np.average(GWAL_cindex), np.average(LWGAL_cindex)

        print('###############Output the optimal parameters######################')
        print("GWAL_model (The best bw):", bw[np.argmax(cindex_par[0, :]) % len(bw)])
        print("LWGAL_model (The best lamda):", lamda[np.argmax(cindex_par[1, :]) // len(bw)],
              "(The best bw):", bw[np.argmax(cindex_par[1, :]) % len(bw)])

        print('###############Start conducting the test set test########################')
        Cox_coef_loss, AR_coef_loss, AL_coef_loss, gAL_coef_loss, GWAL_coef_loss, LWGAL_coef_loss \
            = [], [], [], [], [], []
        Cox_coef, AR_coef, AL_coef, gAL_coef, GWAL_coef = [], [], [], [], []
        AL_select, gAL_select, GWAL_select, LWGAL_select = [], [], [], []
        AR_intercept, AL_intercept, gAL_intercept, GWAL_intercept = [], [], [], []

        ########################LWGAL#############################
        gwr_bw = bw[np.argmax(cindex_par[1, :]) % len(bw)]
        Aerfa = Kernel(distance_matrix, gwr_bw, dataset, 'bi-square')
        LWGAL_model = LWGAL(lamd=lamda[np.argmax(cindex_par[1, :]) // len(bw)], tol=0.001, max_iter=1000, threshold=threshold, verbose=False)
        LWGAL_model.fit(X_train_o, y_train_o, Aerfa, datasets, p, M)

        if check_hyperparameter(lamda[np.argmax(cindex_par[1, :]) // len(bw)], gwr_bw, lamda, bw):
            continue

        for loc in range(len(dataset)):
            aerfa = Aerfa.iloc[:, loc]
            XX_test, yy_test = \
                X_test_o[list((X_test_o[:, 0] == dataset[loc])), :][:, 1:].astype(np.float64), \
                    y_test_o[list((X_test_o[:, 0] == dataset[loc]))]
            if len(XX_test) == 0:
                continue
            ####################################LWGAL####################################
            y_LWGAL = LWGAL_model.predict(add_intercept(XX_test), loc, p).flatten()
            C_index_LWGAL[eff, loc] = concordance_index(yy_test[:, 1], y_LWGAL, event_observed=yy_test[:, 0])

        ########################lasso+GWR###############################
        gwr_bw = bw[np.argmax(cindex_par[0, :]) % len(bw)]
        Aerfa = Kernel(distance_matrix, gwr_bw, dataset, 'bi-square')
        for loc in range(len(dataset)):
            aerfa = Aerfa.iloc[:, loc]

            XX_train, yy_train = \
                X_train_o[list((X_train_o[:, 0] == dataset[loc])), :][:, 1:].astype(np.float64), \
                y_train_o[list((X_train_o[:, 0] == dataset[loc]))]

            XX_test, yy_test = \
                X_test_o[list((X_test_o[:, 0] == dataset[loc])), :][:, 1:].astype(np.float64), \
                y_test_o[list((X_test_o[:, 0] == dataset[loc]))]
            if len(XX_test) == 0:
                continue

            ####################################GWAL##############################
            coef, intercept = GWAL(X_train_o, y_train_o, aerfa, threshold)
            GWAL_coef.append(coef)
            GWAL_intercept.append(intercept)

            W = np.hstack((coef, intercept))
            y_GWAL = np.exp(add_intercept(XX_test) @ W)
            C_index_GWAL[eff, loc] = concordance_index(yy_test[:, 1], y_GWAL, event_observed=yy_test[:, 0])

            #################################AL################################
            coef, intercept = AL(XX_train, yy_train, threshold)
            AL_coef.append(coef)
            AL_intercept.append(intercept)

            W = np.hstack((coef, intercept))
            y_AL = np.exp(add_intercept(XX_test) @ W)
            C_index_AL[eff, loc] = concordance_index(yy_test[:, 1], y_AL, event_observed=yy_test[:, 0])

            #################################AR#################################
            coef, intercept = AR(XX_train, yy_train)
            AR_coef.append(coef)
            AR_intercept.append(intercept)

            W = np.hstack((coef, intercept))
            y_AR = np.exp(add_intercept(XX_test) @ W)
            C_index_AR[eff, loc] = concordance_index(yy_test[:, 1], y_AR, event_observed=yy_test[:, 0])

            #################################RSF######################################
            rsf_model = RSF(XX_train, yy_train)
            y_rsf = rsf_model.predict(XX_test)
            C_index_RSF[eff, loc] = concordance_index(yy_test[:, 1], -y_rsf, event_observed=yy_test[:, 0])

            #################################DS########################################
            ds_model = DS(XX_train, yy_train)
            y_ds = ds_model.predict_risk(XX_test.astype('float32')).flatten()
            C_index_DS[eff, loc] = concordance_index(yy_test[:, 1], -y_ds, event_observed=yy_test[:, 0])

            ##################################Cox######################################
            estimator = CoxPHSurvivalAnalysis(tol=0.001, n_iter=1000, alpha=10)
            aux = [(e1, e2) for e1, e2 in yy_train]
            y2 = np.array(aux, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
            estimator.fit(XX_train, y2)
            Cox_coef.append(estimator.coef_)

            y_cox = estimator.predict(XX_test)
            C_index_Cox[eff, loc] = concordance_index(yy_test[:, 1], -y_cox, event_observed=yy_test[:, 0])

        X_train_o, X_test_o = X_train_o[:, 1:].astype(np.float64), X_test_o[:, 1:].astype(np.float64)
        #################################gAL################################
        coef, intercept = AL(X_train_o, y_train_o, threshold)
        gAL_coef.append(coef)
        gAL_intercept.append(intercept)

        W = np.hstack((coef, intercept))
        y_gAL = np.exp(add_intercept(X_test_o) @ W)
        C_index_gAL[eff, :] = concordance_index(y_test_o[:, 1], y_gAL, event_observed=y_test_o[:, 0])

        print('C_index_Ridge', np.average(C_index_AR[eff, :]))
        print('C_index_Lasso', np.average(C_index_AL[eff, :]))
        print('C_index_gLasso', np.average(C_index_gAL[eff, :]))
        print('C_index_GWAL', np.average(C_index_GWAL[eff, :]))
        print('C_index_LWGAL', np.average(C_index_LWGAL[eff, :]))
        print('C_index_Cox', np.average(C_index_Cox[eff, :]))
        print('C_index_RSF', np.average(C_index_RSF[eff, :]))
        print('C_index_DS', np.average(C_index_DS[eff, :]))

        print('***********Coefficient estimation error***********')
        Cox_coef_loss.append(Criteria().estimation_process(np.array(Cox_coef), np.zeros(wm+sm), beta))

        AR_coef_loss.append(Criteria().estimation_process(np.array(AR_coef), np.array(AR_intercept), beta))

        AL_coef_loss.append(Criteria().estimation_process(np.array(AL_coef), np.array(AL_intercept), beta))
        AL_select.append(Criteria().selection_process(np.array(AL_coef), beta))

        gAL_coef_loss.append(Criteria().estimation_process(np.array(gAL_coef), np.array(gAL_intercept), beta))
        gAL_select.append(Criteria().selection_process(np.tile(np.array(gAL_coef), (wm+sm,1)), beta))

        GWAL_coef_loss.append(Criteria().estimation_process(np.array(GWAL_coef), np.array(GWAL_intercept), beta))
        GWAL_select.append(Criteria().selection_process(np.array(GWAL_coef), beta))

        LWGAL_coef_loss.append(Criteria().estimation_process(np.array(LWGAL_model.coef_.reshape((M, p))), np.array(LWGAL_model.intercept_), beta))
        LWGAL_select.append(Criteria().selection_process(np.array(LWGAL_model.coef_.reshape((M, p))), beta))

        print("Simulation", eff + 1, "Coxcl, ARcl, ALcl, gALcl, GWALcl, LWGALcl", end='')
        print(Cox_coef_loss, AR_coef_loss, AL_coef_loss, gAL_coef_loss, GWAL_coef_loss,LWGAL_coef_loss)

        print("Selection")
        print(AL_select, gAL_select, GWAL_select,LWGAL_select)

        coef_MAB_loss[0, eff] = Cox_coef_loss[0][0]
        coef_MAB_loss[1, eff] = AR_coef_loss[0][0]
        coef_MAB_loss[2, eff] = AL_coef_loss[0][0]
        coef_MAB_loss[3, eff] = gAL_coef_loss[0][0]
        coef_MAB_loss[4, eff] = GWAL_coef_loss[0][0]
        coef_MAB_loss[5, eff] = LWGAL_coef_loss[0][0]

        coef_MSE_loss[0, eff] = Cox_coef_loss[0][1]
        coef_MSE_loss[1, eff] = AR_coef_loss[0][1]
        coef_MSE_loss[2, eff] = AL_coef_loss[0][1]
        coef_MSE_loss[3, eff] = gAL_coef_loss[0][1]
        coef_MSE_loss[4, eff] = GWAL_coef_loss[0][1]
        coef_MSE_loss[5, eff] = LWGAL_coef_loss[0][1]

        TP[0, eff] = AL_select[0][0]
        TP[1, eff] = gAL_select[0][0]
        TP[2, eff] = GWAL_select[0][0]
        TP[3, eff] = LWGAL_select[0][0]

        Model_Size[0, eff] = AL_select[0][1]
        Model_Size[1, eff] = gAL_select[0][1]
        Model_Size[2, eff] = GWAL_select[0][1]
        Model_Size[3, eff] = LWGAL_select[0][1]

        Cindex[0, eff] = np.average(C_index_AR[eff, :])
        Cindex[1, eff] = np.average(C_index_AL[eff, :])
        Cindex[2, eff] = np.average(C_index_gAL[eff, :])
        Cindex[3, eff] = np.average(C_index_GWAL[eff, :])
        Cindex[4, eff] = np.average(C_index_LWGAL[eff, :])
        Cindex[5, eff] = np.average(C_index_Cox[eff, :])
        Cindex[6, eff] = np.average(C_index_RSF[eff, :])
        Cindex[7, eff] = np.average(C_index_DS[eff, :])

        print('Coefficient saved.....\n\n\n\n\n\n\n')
        AR_coef, AL_coef, gAL_coef, GWAL_coef = np.array(AR_coef), np.array(AL_coef), np.array(gAL_coef), np.array(GWAL_coef)
        AR_intercept, AL_intercept, gAL_intercept, GWAL_intercept =\
            np.array(AR_intercept), np.array(AL_intercept), np.array(gAL_intercept), np.array(GWAL_intercept)
        LWGAL_coef, LWGAL_intercept = np.array(LWGAL_model.coef_.reshape((M, p))), np.array(
            LWGAL_model.intercept_)
        Cox_fin = np.array(Cox_coef)
        AR_fin = np.hstack((AR_coef, AR_intercept.reshape(-1, 1)))
        AL_fin = np.hstack((AL_coef, AL_intercept.reshape(-1, 1)))
        gAL_fin = np.hstack((gAL_coef, gAL_intercept.reshape(-1, 1)))
        GWAL_fin = np.hstack((GWAL_coef, GWAL_intercept.reshape(-1, 1)))
        LWGAL_fin = np.hstack((LWGAL_coef, LWGAL_intercept.reshape(-1, 1)))

        output_dir = 'estimated_coef/' + 'repeat_' + str(eff+1)
        if output_dir[-1] != "/":
            output_dir += "/"
        if not os.path.isdir(output_dir):
            print("Directory doesn't exist, creating it")
            os.mkdir(output_dir)
        pd.DataFrame(Cox_fin).to_excel(output_dir + "Coef_Cox" + ".xlsx")
        pd.DataFrame(AR_fin).to_excel(output_dir + "Coef_AR" + ".xlsx")
        pd.DataFrame(AL_fin).to_excel(output_dir + "Coef_AL" + ".xlsx")
        pd.DataFrame(gAL_fin).to_excel(output_dir + "Coef_gAL" + ".xlsx")
        pd.DataFrame(GWAL_fin).to_excel(output_dir + "Coef_GWAL" + ".xlsx")
        pd.DataFrame(LWGAL_fin).to_excel(output_dir + "Coef_LWGAL" + ".xlsx")
        eff += 1
    except Exception as e:
        pass
    continue

''''''''''''''''''''



''' evaluation result '''
###############################Coefficient estimation results are saved####################################
coef_MAB_average = coef_MAB_loss.mean(axis=1)
coef_MAB_std = coef_MAB_loss.std(axis=1)
coef_MSE_average = coef_MSE_loss.mean(axis=1)
coef_MSE_std = coef_MSE_loss.std(axis=1)
TP_average = TP.mean(axis=1)
TP_std = TP.std(axis=1)
Model_Size_average = Model_Size.mean(axis=1)
Model_Size_std = Model_Size.std(axis=1)
Cindex_average = Cindex.mean(axis=1)
Cindex_std = Cindex.std(axis=1)

output_dir = 'evaluation_result/'
if output_dir[-1] != "/":
    output_dir += "/"
if not os.path.isdir(output_dir):
    print("Directory doesn't exist, creating it")
    os.mkdir(output_dir)

result = {}
result['Cox_coef_average_MAB_loss(var)'] = [
    str(round(coef_MAB_average[0], 6)) + ' (' + str(round(coef_MAB_std[0], 6)) + ')']
result['AR_coef_average_MAB_loss(var)'] = [
    str(round(coef_MAB_average[1], 6)) + ' (' + str(round(coef_MAB_std[1], 6)) + ')']
result['AL_coef_average_MAB_loss(var)'] = [
    str(round(coef_MAB_average[2], 6)) + ' (' + str(round(coef_MAB_std[2], 6)) + ')']
result['gAL_coef_average_MAB_loss(var)'] = [
    str(round(coef_MAB_average[3], 6)) + ' (' + str(round(coef_MAB_std[3], 6)) + ')']
result['GWAL_coef_average_MAB_loss(var)'] = [
    str(round(coef_MAB_average[4], 6)) + ' (' + str(round(coef_MAB_std[4], 6)) + ')']
result['LWGAL_coef_average_MAB_loss(var)'] = [
    str(round(coef_MAB_average[5], 6)) + ' (' + str(round(coef_MAB_std[5], 6)) + ')']

result['Cox_coef_average_MSE_loss(var)'] = [
    str(round(coef_MSE_average[0], 7)) + ' (' + str(round(coef_MSE_std[0], 7)) + ')']
result['AR_coef_average_MSE_loss(var)'] = [
    str(round(coef_MSE_average[1], 7)) + ' (' + str(round(coef_MSE_std[1], 7)) + ')']
result['AL_coef_average_MSE_loss(var)'] = [
    str(round(coef_MSE_average[2], 7)) + ' (' + str(round(coef_MSE_std[2], 7)) + ')']
result['gAL_coef_average_MSE_loss(var)'] = [
    str(round(coef_MSE_average[3], 7)) + ' (' + str(round(coef_MSE_std[3], 7)) + ')']
result['GWAL_coef_average_MSE_loss(var)'] = [
    str(round(coef_MSE_average[4], 7)) + ' (' + str(round(coef_MSE_std[4], 7)) + ')']
result['LWGAL_coef_average_MSE_loss(var)'] = [
    str(round(coef_MSE_average[5], 7)) + ' (' + str(round(coef_MSE_std[5], 7)) + ')']

result['AL_tp(var)'] = [str(round(TP_average[0], 3)) + ' (' + str(round(TP_std[0], 3)) + ')']
result['gAL_tp(var)'] = [str(round(TP_average[1], 3)) + ' (' + str(round(TP_std[1], 3)) + ')']
result['GWAL_tp(var)'] = [str(round(TP_average[2], 3)) + ' (' + str(round(TP_std[2], 3)) + ')']
result['LWGAL_tp(var)'] = [str(round(TP_average[3], 3)) + ' (' + str(round(TP_std[3], 3)) + ')']

result['AL_model_size(var)'] = [str(round(Model_Size_average[0], 3)) + ' (' + str(round(Model_Size_std[0], 3)) + ')']
result['gAL_model_size(var)'] = [str(round(Model_Size_average[1], 3)) + ' (' + str(round(Model_Size_std[1], 3)) + ')']
result['GWAL_model_size(var)'] = [str(round(Model_Size_average[2], 3)) + ' (' + str(round(Model_Size_std[2], 3)) + ')']
result['LWGAL_model_size(var)'] = [str(round(Model_Size_average[3], 3)) + ' (' + str(round(Model_Size_std[3], 3)) + ')']


result['AR_average_Cindex(std)'] = [str(round(Cindex_average[0], 3)) + ' (' + str(round(Cindex_std[0], 3)) + ')']
result['AL_average_Cindex(std)'] = [str(round(Cindex_average[1], 3)) + ' (' + str(round(Cindex_std[1], 3)) + ')']
result['gAL_average_Cindex(std)'] = [str(round(Cindex_average[2], 3)) + ' (' + str(round(Cindex_std[2], 3)) + ')']
result['GWAL_average_Cindex(std)'] = [str(round(Cindex_average[3], 3)) + ' (' + str(round(Cindex_std[3], 3)) + ')']
result['LWGAL_average_Cindex(std)'] = [str(round(Cindex_average[4], 3)) + ' (' + str(round(Cindex_std[4], 3)) + ')']
result['Cox_average_Cindex(std)'] = [str(round(Cindex_average[5], 3)) + ' (' + str(round(Cindex_std[5], 3)) + ')']
result['RSF_average_Cindex(std)'] = [str(round(Cindex_average[6], 3)) + ' (' + str(round(Cindex_std[6], 3)) + ')']
result['DS_average_Cindex(std)'] = [str(round(Cindex_average[7], 3)) + ' (' + str(round(Cindex_std[7], 3)) + ')']

save_files(pd.DataFrame(result), output_dir + 'wm, sm, M=' + str(wm) + ',' + str(sm) + ',' + str(M) + '_pc, pg, indp=' + str(pc)
    + ',' + str(pg) + ',' + str(indp) + '_correlation=' + correlation_structure + '_relationship='
    + relationship_structure + '_average_n_m, std=' + str(n) + ',' + str(ns))
''''''''''''''''''