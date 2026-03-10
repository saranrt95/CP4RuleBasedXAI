import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from rule_generation_utils import *


def GetPredictionRegions(y_calib, scores0, scores1, scores0ts, scores1ts, epsilon, n_c):

    scores_calib = np.where(y_calib == 0, scores0, scores1)

    scores_calib_sorted = np.sort(scores_calib)

    qlevel = np.ceil((n_c + 1) * (1 - epsilon)) / n_c
    if n_c < 100: qlevel = 1 - epsilon
    #print("qlevel: ", qlevel)
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
    varErr = np.zeros(n_eps)

    avgErr_singleton = np.zeros(n_eps)
    varErr_singleton = np.zeros(n_eps)

    empty = np.zeros(n_eps)
    var_empty = np.zeros(n_eps)

    singleton = np.zeros(n_eps)
    var_singleton = np.zeros(n_eps)

    double = np.zeros(n_eps)
    var_double = np.zeros(n_eps)

    avgSize = np.zeros(n_eps)
    varSize = np.zeros(n_eps)

    for i, epsilon in enumerate(epsilonrange):

        C_all, _, _, C_size = GetPredictionRegions(
            Ycal, tau0cal, tau1cal, tau0ts, tau1ts, epsilon, n_c
        )
        err_vec = ((Yts != C_all) & (C_all != 2)).astype(float)
        err_singleton_vec = ((Yts != C_all) & (C_size == 1)).astype(float)

        empty_vec = (C_size == 0).astype(float)
        singleton_vec = (C_size == 1).astype(float)
        double_vec = (C_size == 2).astype(float)

        size_vec = C_size.astype(float)

        avgErr[i] = np.mean(err_vec)
        avgErr_singleton[i] = np.mean(err_singleton_vec)
        empty[i] = np.mean(empty_vec)
        singleton[i] = np.mean(singleton_vec)
        double[i] = np.mean(double_vec)
        avgSize[i] = np.mean(size_vec)

        varErr[i] = np.var(err_vec)
        varErr_singleton[i] = np.var(err_singleton_vec)
        var_empty[i] = np.var(empty_vec)
        var_singleton[i] = np.var(singleton_vec)
        var_double[i] = np.var(double_vec)
        varSize[i] = np.var(size_vec)

        with open(f"{res_path}/metrics.csv","a") as ff:
            ff.write(
                f"{score_fn},{epsilon},"
                f"{avgErr[i]},{varErr[i]},"
                f"{singleton[i]},{var_singleton[i]},"
                f"{double[i]},{var_double[i]},"
                f"{empty[i]},{var_empty[i]},"
                f"{avgSize[i]},{varSize[i]},"
                f"{avgErr_singleton[i]},{varErr_singleton[i]}\n"
            )

    return (
        avgErr, varErr,
        avgErr_singleton, varErr_singleton,
        empty, var_empty,
        singleton, var_singleton,
        double, var_double,
        avgSize, varSize
    )


