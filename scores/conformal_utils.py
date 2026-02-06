import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from rule_generation_utils import *


def GetPredictionRegions(y_calib, scores0, scores1, scores0ts, scores1ts, epsilon, n_c):

    scores_calib = np.where(y_calib == 0, scores0, scores1)

    scores_calib_sorted = np.sort(scores_calib)

    qlevel = np.ceil((n_c + 1) * (1 - epsilon)) / n_c
    s_epsilon = np.quantile(scores_calib_sorted, qlevel)

    n_test = scores0ts.shape[0]
    C_all = np.full(n_test, np.nan)
    C_size = np.full(n_test, np.nan)

    for r in range(n_test):
        tau0_r = scores0ts[r]
        tau1_r = scores1ts[r]
        if tau0_r <= s_epsilon and tau1_r > s_epsilon:
            C_all[r] = 0
            C_size[r] = 1
        elif tau0_r > s_epsilon and tau1_r <= s_epsilon:
            C_all[r] = 1
            C_size[r] = 1
        elif tau0_r <= s_epsilon and tau1_r <= s_epsilon:
            C_all[r] = 2
            C_size[r] = 2
        elif tau0_r > s_epsilon and tau1_r > s_epsilon:
            C_all[r] = 3
            C_size[r] = 0
        else:
            continue   
    return C_all, s_epsilon, scores_calib, C_size




def evaluate_conformal(Ycal, tau0cal, tau1cal, n_c, epsilonrange, Yts, tau0ts, tau1ts, res_path, score_fn):
    n_eps = len(epsilonrange)
    avgErr = np.zeros(n_eps)
    avgErr_singleton = np.zeros(n_eps)

    empty = np.zeros(n_eps)
    singleton = np.zeros(n_eps)
    double = np.zeros(n_eps)

    avgSize = np.zeros(n_eps)
    for i, epsilon in enumerate(epsilonrange):
        # Compute prediction regions
        C_all, _, _, C_size = GetPredictionRegions(Ycal, tau0cal, tau1cal, tau0ts, tau1ts, epsilon, n_c)
        
        n_err = np.sum((Yts != C_all) & (C_all != 2))
        n_err_singleton = np.sum((Yts!=C_all) & (C_size == 1))

        avgErr[i] = n_err / len(Yts)
        avgErr_singleton[i] = n_err_singleton/len(Yts)    
        empty[i] = np.sum(C_size == 0) / len(Yts)
        singleton[i] = np.sum(C_size == 1) / len(Yts)
        double[i] = np.sum(C_size == 2) / len(Yts)
        avgSize[i] = np.sum(C_size)/len(Yts)

        with open(f"{res_path}/metrics.csv","a") as ff:
            ff.write(f"{score_fn},{epsilon},{avgErr[i]},{singleton[i]},{double[i]},{empty[i]},{avgSize[i]},{avgErr_singleton[i]}\n")
    return avgErr, avgErr_singleton, empty, singleton, double, avgSize


