import numpy as np

### MARGIN-based ####
def m_estimate_proba(model, X, y_train, m = 0.0):
    '''
    n_c: number of instances from class j that fall in the same leaf as instance i
    m: parameter in [0,inf] --> larger smoother
    p_cj: prior probability for class j
    '''
    dt_tree = model.tree_

    #y_train = model.classes_ 
    class_counts = np.bincount(y_train)
    p_c = class_counts / class_counts.sum()
    idx_leaves = model.apply(X)
    probabilities = []
    for node in idx_leaves:
        # Get counts for this leaf
        n_c = dt_tree.value[node][0] # raw counts per class
        n = n_c.sum()
        # m-estimate
        p = (n_c + m * p_c) / (n + m)
        probabilities.append(p)

    return np.vstack(probabilities)

def margin_based_score(model, X, y_train, y, mode = "m-estimate"):
    if mode == "standard":
        pr = model.predict_proba(X)
    elif mode == "m-estimate":
        pr = m_estimate_proba(model, X, y_train, m = 2.0)
    #print(pr)
    if y == 0:
        score = pr[:,0] - pr[:,1]
    elif y == 1:
        score = pr[:, 1] - pr[:,0]
    return ((-score +1)/2) # use - to get a nonconformity
