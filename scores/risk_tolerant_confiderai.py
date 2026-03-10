import numpy as np


from rule_generation_utils import *

#### RT-CONFIDERAI SCORE ######


def compute_centerbased_gamma_multid(rule_limits, X_r, distance="L1", normalization="tanh", p=1):
    """
    rule_limits: array-like of shape (2*d,) -> [l1,u1,l2,u2,...]
    X_r: array-like of shape (d,)
    """

    rule_limits = np.asarray(rule_limits)
    #print(rule_limits.shape)
    X_r = np.asarray(X_r)

    d = X_r.shape[0]

    # split lower / upper bounds
    lower = rule_limits[0::2]
    upper = rule_limits[1::2]

    # centers and half-widths
    centers = (lower + upper) / 2.0
    half_widths = np.abs(upper - lower) / 2.0

    # avoid division by zero
    half_widths = np.clip(half_widths, 1e-12, None)

    gamma_vec = (np.abs(X_r - centers) / half_widths)**p

    if distance == "L2":
        tau_geom = np.sqrt(np.mean(gamma_vec ** 2))
    elif distance == "L1":
        tau_geom = np.mean(gamma_vec)
    elif distance == "Linf":
         tau_geom = np.max(gamma_vec, axis = 0)
    else:
        raise ValueError("distance must be 'L1', 'L2' or 'Linf' ")

    if normalization == "decay":
        tau_geom = tau_geom/(1+np.abs(tau_geom))
    elif normalization == "tanh":
        tau_geom = np.tanh(tau_geom)
    elif normalization == "sigmoid":
        tau_geom = 2*(1/(1+np.exp(-tau_geom)) - 0.5)
    elif normalization == "atan":
        tau_geom = (2/np.pi) *  np.atan((np.pi/2)*tau_geom)
    else:
        raise ValueError("normalization function must be one of: 'decay','tanh', 'sigmoid', 'atan'")

    return tau_geom


def compute_class_similarity_terms(rulesim, changeclsidx, y):
    """
    Compute within-class similarity W_y and cross-class similarity C_y
    for y = 0 and y = 1
    """

    # define the rule index sets
    if y == 0:
        R_y = np.arange(0, changeclsidx-1)
        R_not_y = np.arange(changeclsidx-1, rulesim.shape[0])
    elif y == 1:
        R_y = np.arange(changeclsidx-1, rulesim.shape[0])
        R_not_y = np.arange(0, changeclsidx-1)

    # ---- WITHIN-CLASS SIMILARITY ----
    def within_class_similarity(R):
        if len(R) <= 1:
            return 0.0   # no pairwise similarities possible
        sub = rulesim[np.ix_(R, R)]   # |R| x |R| block
        # remove diagonal
        mask = ~np.eye(len(R), dtype=bool)
        return sub[mask].mean()

    W_y = within_class_similarity(R_y)

    # ---- CROSS-CLASS SIMILARITY ----
    def cross_class_similarity(Ra, Rb):
        if len(Ra)==0 or len(Rb)==0:
            return 0.0
        sub = rulesim[np.ix_(Ra, Rb)]   # |Ra| x |Rb| block
        return sub.mean()

    C_y = cross_class_similarity(R_y, R_not_y)

    return W_y, C_y


def risk_tolerant_confiderai_score(X_r, rulesim, rule_limits, changeclsidx, y, relevance, q_y, q_not_y, distance = "L1", p=1, norm_function = "tanh", aggregation = "geometric_mean", beta = None, use_relevance = True, use_sim = True):

    if y == 0:
        idxrules = range(changeclsidx-1)#verified[verified < changeclsidx-1]
    elif y == 1:
        idxrules = range(changeclsidx-1, rulesim.shape[0])#verified[verified >= changeclsidx-1]
    
    score_all_rules = []

    gamma_all_rules = []#1
    sim_term_tot = []#1

    for r in idxrules:#idxrules:
        gamma = compute_centerbased_gamma_multid(rule_limits[r], X_r, distance = distance, p=p, normalization= norm_function)#compute_borderbased_gamma(rule_limits[r], X_r[0], X_r[1])#compute_centerbased_gamma(rule_limits[r], X_r[0], X_r[1])
        gamma_all_rules.append(gamma)

        # apply relevance
        if use_relevance:
            gamma *= (1 - relevance[r])

        #tauprod *= tau
        score_all_rules.append(gamma)

    if aggregation == "geometric_mean":
        gamma_arr = np.array(gamma_all_rules)
        gamma_arr = np.clip(gamma_arr, 1e-15, 1.0)
        final_gamma = float(np.exp(np.mean(np.log(gamma_arr))))

        # geometric mean definition (to avoid collapse towards 0)
        score_arr = np.array(score_all_rules)
        score_arr = np.clip(score_arr, 1e-15, 1.0)
        if use_sim:
            score = float(np.exp(np.mean(np.log(score_arr)))) * 0.5 * (1+q_not_y-q_y)
        else: 
            score = float(np.exp(np.mean(np.log(score_arr))))
    
    elif aggregation == "sum":
        gamma_arr = np.sum(gamma_all_rules)
        final_gamma = 1 - np.clip(gamma_arr, 0, 1)

        score_arr = np.sum(score_all_rules)
        score = 1 - np.clip(score_arr, 0, 1)
        if use_sim:
            score = score * 0.5 * (1+q_not_y-q_y)
    
    elif aggregation == "weighted_average":
        #print("gamma_all_rules: ", gamma_all_rules)
        weights = np.exp(-beta * np.array(gamma_all_rules))
        
        score = np.sum(weights * np.array(gamma_all_rules)) / np.sum(weights)
        final_gamma = score
        if use_sim:
            score = score * 0.5 * (1+q_not_y-q_y)

    #print("s(x,y) = ", tauprod)

    return score, final_gamma, 0.5 * (1+q_not_y-q_y)



def compute_confiderai_score(X, rulesim, rule_limits, changeclsidx, y, relevance, distance = "L1", p = 1,norm_function = "tanh", aggregation = "geometric_mean", beta = None, use_relevance = True, use_sim = True):

    N_points = X.shape[0]
    N_rules = rulesim.shape[0]

    q_y, q_not_y = compute_class_similarity_terms(rulesim, changeclsidx, y)

    # Compute score for each point
    tau = np.empty(N_points)
    gamma = np.empty(N_points)
    simterm = np.empty(N_points)
    for i in range(N_points):
        tau[i], gamma[i], simterm[i] = risk_tolerant_confiderai_score(X[i], rulesim, rule_limits, changeclsidx, y, relevance, q_y, q_not_y,  distance = distance, p = p, norm_function = norm_function, aggregation = aggregation, beta = beta, use_relevance=use_relevance, use_sim = use_sim)
    
    return tau, gamma, simterm





