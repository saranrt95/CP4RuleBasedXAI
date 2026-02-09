import numpy as np


def confiderai_score(X_r, S_row, rulesim, rule_limits, changeclsidx, y, relevance):
    verified = np.where(S_row == 1)[0]
    
    if y == 0:
        verified = verified[verified < changeclsidx-1]
    elif y == 1:
        verified = verified[verified >= changeclsidx-1]
    
    tauprod = 1.0
    if len(verified) == 0: 

        # the point does not satisfy any rule from candidate class
        return 1.0, np.nan, np.nan # soft_fallback_tau(X_r, y, rulesim, rule_limits, changeclsidx, relevance) # 1.0, np.nan, np.nan # 

    for r in verified:

        bounds = rule_limits[r]          # shape (2*d,)
        lower = bounds[0::2]             # (d,)
        upper = bounds[1::2]             # (d,)

        width = np.abs(upper - lower)
        width = np.clip(width, 1e-12, None)   # avoid divide-by-zero

        dist_lower = np.abs(X_r - lower)
        dist_upper = np.abs(upper - X_r)

        gamma_per_dim = 1.0 - np.minimum(dist_lower, dist_upper) / width
        #print(gamma_per_dim)
        tau_geom = np.mean(gamma_per_dim)
        #print("gamma: ", tau_geom)
        # similarity terms to tune the geometrical tau
        if y == 0:
            same_idx = [i for i in range(changeclsidx-1) if i != r]
            opp_idx = [i for i in range(changeclsidx-1, rulesim.shape[1])]
        else:  # y == 1
            same_idx = [i for i in range(changeclsidx-1, rulesim.shape[1]) if i != r]
            opp_idx = [i for i in range(changeclsidx-1)]
        
        avg_sim_same = np.nanmean(rulesim[r, same_idx]) if same_idx else 0
        avg_sim_opposite = np.nanmean(rulesim[r, opp_idx]) if opp_idx else 0
        
        sim_term = avg_sim_opposite - avg_sim_same
        #print("sim_term: ", sim_term)
        # modified tau
        tau = 0.5 * tau_geom * (1 + sim_term)
        # apply relevance
        tau *= (1 - relevance[r])
        tauprod *= tau
    return tauprod, tau_geom, (1+sim_term)/2

def compute_dataset_score(X, rulesim, rule_limits, changeclsidx, y, relevance):

    N_points = X.shape[0]
    N_rules = rulesim.shape[0]
    #print(rule_limits)
    # Initialize S
    S = np.zeros((N_points, N_rules))
    
    # Compute which rules are verified for each point
    for i in range(N_points):
        for j in range(N_rules):
            bounds = rule_limits[j]
            lower = bounds[0::2]
            upper = bounds[1::2]
            S[i, j] = float(np.all((lower <= X[i]) & (X[i] <= upper)))


    # Compute tau for each point
    tau = np.empty(N_points)
    gamma = np.empty(N_points)
    simterm = np.empty(N_points)
    for i in range(N_points):
        tau[i], gamma[i], simterm[i] = confiderai_score(X[i], S[i], rulesim, rule_limits, changeclsidx, y, relevance)
    
    return tau, S, gamma, simterm


