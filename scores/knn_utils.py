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

    #if len(X_same) < K or len(X_diff) < K:
    #    return np.full(X_rule.shape[0], 0.5)

    knn_same = NearestNeighbors(n_neighbors=K).fit(X_same)
    knn_diff = NearestNeighbors(n_neighbors=K).fit(X_diff)

    scores = np.zeros(X_rule.shape[0])

    for i, x in enumerate(X_rule):
        x = x.reshape(1, -1)
        d_same, _ = knn_same.kneighbors(x)
        d_diff, _ = knn_diff.kneighbors(x)

        den = d_diff.sum()
        score = np.inf if den == 0 else d_same.sum() / den
        scores[i] = 1.0 / (1.0 + np.exp(-score))

    return scores

def KNN_Score(X, Y, ruleset_path, featurelabels, outputlabel, cls0label, cls1label, K=10):

    # rearrange calibration and test data into dataframes (for feature names availability)
    X_cal = pd.DataFrame(X, columns=featurelabels)
    Y_cal = pd.DataFrame(Y, columns=[outputlabel])

    # group points based on rules satisfaction
    Xcal_rules, Ycal_rules, cal_idx, _ = SatisfiedRules(X_cal, Y_cal, ruleset_path, outputlabel)

    knnscores0_cal = []
    knnscores1_cal = []

    n_cal = len(X_cal)

    scores0_cal_knn = np.full(n_cal, np.nan)
    scores1_cal_knn = np.full(n_cal, np.nan)

    for r in range(len(Xcal_rules)):
        knnscores0_cal.append(compute_knn_score_rulewise(Xcal_rules[r], X_cal, Y_cal, cls0label, K = K))
        knnscores1_cal.append(compute_knn_score_rulewise(Xcal_rules[r], X_cal, Y_cal, cls1label, K = K))

    for r in range(len(cal_idx)):
        idx = cal_idx[r]
        scores0_cal_knn[idx] = knnscores0_cal[r]
        scores1_cal_knn[idx] = knnscores1_cal[r]
    max0 = np.nanmax(scores0_cal_knn)
    max1 = np.nanmax(scores1_cal_knn)

    unverified = np.isnan(scores0_cal_knn)

    scores0_cal_knn[unverified] = max0
    scores1_cal_knn[unverified] = max1

    #cal_idx_all = np.hstack(cal_idx)
    #order = np.argsort(cal_idx_all)
    #scores0_cal_knn = np.hstack(knnscores0_cal)[cal_idx_all]
    #scores1_cal_knn = np.hstack(knnscores1_cal)[cal_idx_all]

    return scores0_cal_knn, scores1_cal_knn#, Xcal_rules, Ycal_rules, cal_idx, order 


def get_prediction_regions_kNN(ycal_rules, test_idx, X_test, scores0_cal, scores1_cal, scores0_ts, scores1_ts, epsilon):

    R = len(scores0_cal)
    #n_test = sum(len(idx) for idx in test_idx)
    n_test = len(X_test)

    C_all = np.full(n_test, 2.0)
    C_size = np.full(n_test, 0.0)
    s_epsilon = np.full(R, np.nan)

    for r in range(R):

        if len(test_idx[r]) == 0 or len(scores0_cal[r]) == 0:
            continue

        y_calib = ycal_rules[r]

        C_eps_r, s_eps_r, _, C_size_r = GetPredictionRegions(
            y_calib,
            scores0_cal[r],
            scores1_cal[r],
            scores0_ts[r],
            scores1_ts[r],
            epsilon,
            len(scores0_cal[r]),
            APPROX=True
        )

        idx = test_idx[r]
        C_all[idx] = C_eps_r
        C_size[idx] = C_size_r
        s_epsilon[r] = s_eps_r


    return C_all, s_epsilon, None, C_size