import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from rule_generation_utils import *
from scores.conformal_utils import *
# KNN utils


def SatisfiedRules(X_cal, y_cal, rulesetfile, outputlabels):
    relevances = ComputeRelevances(rulesetfile)          
    sat_rules = GetSatisfiedRules(X_cal, rulesetfile, outputlabels)   

    n_samples, n_rules = sat_rules.shape

    masked = np.where(sat_rules == 1, relevances, -np.inf)
    best_rules = masked.argmax(axis=1)

    has_rule = sat_rules.any(axis=1)
    best_rules[~has_rule] = -1

    Xcal_rules = []
    Ycal_rules = []
    indices_rules = []

    for r in range(n_rules):
        mask = best_rules == r
        Xcal_rules.append(X_cal[mask].values)
        Ycal_rules.append(y_cal[mask].values)
        indices_rules.append(np.where(mask)[0])

    return Xcal_rules, Ycal_rules, indices_rules, best_rules




def compute_knn_score_rulewise(X_rule, X_all, y_all, y_candidate, K=5):

    if X_rule.shape[0] == 0:
        return np.array([])

    X_same = X_all[y_all.values == y_candidate].values
    X_diff = X_all[y_all.values != y_candidate].values

    # drop NaNs
    #X_same = X_same[~np.isnan(X_same).any(axis=1)]
    #X_diff = X_diff[~np.isnan(X_diff).any(axis=1)]

    if len(X_same) < K or len(X_diff) < K:
        return np.full(X_rule.shape[0], 0.5)

    knn_same = NearestNeighbors(n_neighbors=K).fit(X_same)
    knn_diff = NearestNeighbors(n_neighbors=K).fit(X_diff)

    scores = np.zeros(X_rule.shape[0])

    for i, x in enumerate(X_rule):
        x = x.reshape(1, -1)
        d_same, _ = knn_same.kneighbors(x)
        d_diff, _ = knn_diff.kneighbors(x)

        den = d_diff.sum()
        score = np.inf if den == 0 else d_same.sum() / den
        scores[i] = 2* (1.0 / (1.0 + np.exp(-score)) - 0.5)
    #print(scores)
    return scores


def KNN_Score(X, Y, ruleset_path, featurelabels, outputlabel, cls0label, cls1label, K=10, min_samples_rule=20):

    X_cal = pd.DataFrame(X, columns=featurelabels)
    Y_cal = pd.DataFrame(Y, columns=[outputlabel])

    Xcal_rules, Ycal_rules, cal_idx, _ = SatisfiedRules(X_cal, Y_cal, ruleset_path, outputlabel)

    knnscores0_cal = []
    knnscores1_cal = []

    n_cal = len(X_cal)

    scores0_cal_knn = np.full(n_cal, np.nan)
    scores1_cal_knn = np.full(n_cal, np.nan)

    for r in range(len(Xcal_rules)):
        n_points = len(Xcal_rules[r])
        print(f"rule {r}: {n_points} samples")

        if n_points < min_samples_rule:
            knnscores0_cal.append(np.full(n_points, np.nan))
            knnscores1_cal.append(np.full(n_points, np.nan))
            continue

        knnscores0_cal.append(
            compute_knn_score_rulewise(Xcal_rules[r], X_cal, Y_cal, cls0label, K=K)
        )
        knnscores1_cal.append(
            compute_knn_score_rulewise(Xcal_rules[r], X_cal, Y_cal, cls1label, K=K)
        )
    # return lists of arrays
    return Xcal_rules, Ycal_rules, knnscores0_cal, knnscores1_cal #scores0_cal_knn, scores1_cal_knn

def get_prediction_regions_kNN(ycal_rules, scores0_cal, scores1_cal, scores0_ts, scores1_ts, epsilon):

    R = len(scores0_cal)
    #n_test = sum(len(idx) for idx in test_idx)

    C_all = []#np.full(n_test, np.nan)
    C_size = []#np.full(n_test, 0.0)
    s_epsilon = []#np.full(R, np.nan)
    #print("epsilon: ", epsilon)
    for r in range(R):

        if len(scores0_cal[r]) == 0:
            continue

        #y_calib = ycal_rules[r]
       
        C_eps_r, s_eps_r, _, C_size_r = GetPredictionRegions(
            ycal_rules[r],
            scores0_cal[r],
            scores1_cal[r],
            scores0_ts[r],
            scores1_ts[r],
            epsilon,
            len(scores0_cal[r])
        )
        #print(f"r: {r}, s_eps_r: {s_eps_r}")
        #idx = test_idx[r]
        C_all.append(C_eps_r)
        C_size.append(C_size_r)
        s_epsilon.append(s_eps_r)


    return C_all, s_epsilon, None, C_size


def evaluate_conformal_knn(
    Ycal_rules,
    tau0cal_rules,
    tau1cal_rules,
    epsilonrange,
    Yts_rules,
    tau0ts_rules,
    tau1ts_rules,
    res_path,
    score_fn
):

    n_eps = len(epsilonrange)

    avgErr = np.zeros(n_eps)
    stdErr = np.zeros(n_eps)

    avgErr_singleton = np.zeros(n_eps)
    stdErr_singleton = np.zeros(n_eps)

    empty = np.zeros(n_eps)
    std_empty = np.zeros(n_eps)

    singleton = np.zeros(n_eps)
    std_singleton = np.zeros(n_eps)

    double = np.zeros(n_eps)
    std_double = np.zeros(n_eps)

    avgSize = np.zeros(n_eps)
    stdSize = np.zeros(n_eps)

    for i, epsilon in enumerate(epsilonrange):

        C_all, _, _, C_size = get_prediction_regions_kNN(
            Ycal_rules,
            tau0cal_rules,
            tau1cal_rules,
            tau0ts_rules,
            tau1ts_rules,
            epsilon
        )

        err_mean_rules = []
        err_std_rules = []

        err_singleton_mean_rules = []
        err_singleton_std_rules = []

        empty_mean_rules = []
        empty_std_rules = []

        singleton_mean_rules = []
        singleton_std_rules = []

        double_mean_rules = []
        double_std_rules = []

        size_mean_rules = []
        size_std_rules = []

        R = len(C_all)

        for r in range(R):

            if len(C_all[r]) == 0:
                continue

            Yts_r = Yts_rules[r]
            C_r = C_all[r]
            size_r = C_size[r]

            err_vec = ((Yts_r != C_r) & (C_r != 2)).astype(float)
            err_singleton_vec = ((Yts_r != C_r) & (size_r == 1)).astype(float)

            empty_vec = (size_r == 0).astype(float)
            singleton_vec = (size_r == 1).astype(float)
            double_vec = (size_r == 2).astype(float)

            size_vec = size_r.astype(float)

            err_mean_rules.append(np.mean(err_vec))
            err_std_rules.append(np.std(err_vec))

            err_singleton_mean_rules.append(np.mean(err_singleton_vec))
            err_singleton_std_rules.append(np.std(err_singleton_vec))

            empty_mean_rules.append(np.mean(empty_vec))
            empty_std_rules.append(np.std(empty_vec))

            singleton_mean_rules.append(np.mean(singleton_vec))
            singleton_std_rules.append(np.std(singleton_vec))

            double_mean_rules.append(np.mean(double_vec))
            double_std_rules.append(np.std(double_vec))

            size_mean_rules.append(np.mean(size_vec))
            size_std_rules.append(np.std(size_vec))

        avgErr[i] = np.mean(err_mean_rules)
        stdErr[i] = np.mean(err_std_rules)

        avgErr_singleton[i] = np.mean(err_singleton_mean_rules)
        stdErr_singleton[i] = np.mean(err_singleton_std_rules)

        empty[i] = np.mean(empty_mean_rules)
        std_empty[i] = np.mean(empty_std_rules)

        singleton[i] = np.mean(singleton_mean_rules)
        std_singleton[i] = np.mean(singleton_std_rules)

        double[i] = np.mean(double_mean_rules)
        std_double[i] = np.mean(double_std_rules)

        avgSize[i] = np.mean(size_mean_rules)
        stdSize[i] = np.mean(size_std_rules)

        with open(f"{res_path}/metrics.csv", "a") as ff:
            ff.write(
                f"{score_fn},{epsilon},"
                f"{avgErr[i]},{stdErr[i]},"
                f"{singleton[i]},{std_singleton[i]},"
                f"{double[i]},{std_double[i]},"
                f"{empty[i]},{std_empty[i]},"
                f"{avgSize[i]},{stdSize[i]},"
                f"{avgErr_singleton[i]},{stdErr_singleton[i]}\n"
            )

    return (
        avgErr, stdErr,
        avgErr_singleton, stdErr_singleton,
        empty, std_empty,
        singleton, std_singleton,
        double, std_double,
        avgSize, stdSize
    )