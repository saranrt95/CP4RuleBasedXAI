import numpy as np
import pandas as pd
import os


def GeneralizedIoU(parsedruleset, rulesetfile=None, SAVE_RS_VALUES=False, save_path=None, mode = "local"):
    """
    Adapting the concept of generalized IoU to rule similarity
    """

    rule_ids = sorted(parsedruleset["Rule ID"].unique())
    #print(rule_ids)
    n_rules = len(rule_ids)
    N = len(parsedruleset[parsedruleset["Rule ID"]==rule_ids[0]]["Lower"].values)
    #print(f"total number of features D = {N}")

    lowers = [parsedruleset.loc[parsedruleset["Rule ID"]==r, "Lower"].values for r in rule_ids]
    uppers = [parsedruleset.loc[parsedruleset["Rule ID"]==r, "Upper"].values for r in rule_ids]

    
    #domain_diag = np.linalg.norm(global_max - global_min)
    #domain_diag = max(domain_diag, 1e-12)  # avoid div by zero

    sim_matrix = np.zeros((n_rules, n_rules), dtype=float)

    for i, (L1, U1) in enumerate(zip(lowers, uppers)):
        for j, (L2, U2) in enumerate(zip(lowers[i:], uppers[i:]), start=i):
            #print(f"rule {i+1}-rule{j+1}")
            if i == j:
                sim_matrix[i,j] = 1.0
                continue
            #print(L1, U1, L2, U2)
            MaxL = np.maximum(L1, L2)		
            MinU = np.minimum(U1, U2)
            overlap = np.maximum(0.0, MinU - MaxL)
            #print(f"per-dimension overlaps: {overlap}")
            #n_pos = np.sum(overlap > 0)
            #print(f"n_d = {n_pos}")
            #gap = np.maximum(0.0, np.maximum(L2 - U1, L1 - U2))
            #print(f"per-dimension gap: {gap}")
            vol_overlap = np.prod(overlap)
            #print("volume overlap: ", vol_overlap)
            vol1 = np.prod(U1 - L1)
            vol2 = np.prod(U2 - L2)
            volume_union = vol1 + vol2 - vol_overlap
            #print("volume union: ", volume_union)
            IoU = vol_overlap / volume_union if volume_union > 0 else 0.0
            #print("IoU: ", IoU)

            if mode == "global":
                # compute global spans 
                global_min = np.min(np.vstack(lowers), axis=0)
                global_max = np.max(np.vstack(uppers), axis=0)
                
                volume_global = np.prod(global_max-global_min)
                #print("total volume global: ", volume_global)
            if mode == "local":
                minL = np.minimum(L1, L2)
                maxU = np.maximum(U1,U2)
                volume_global = np.prod(maxU-minL)
                #print("total volume local: ", volume_global)

            volume_gap = volume_global-volume_union
            #print("volume gap: ", volume_gap)
            # range  [-1,1]
            GIoU = IoU - volume_gap/volume_global
            #print("GIoU [-1,1]: ", GIoU)

            # map from [-1,1] to [0,1]
            sim_matrix[i,j] = sim_matrix[j,i] = (GIoU+1)/2
            #print("GIoU Similarity [0,1]: ", (GIoU+1)/2)

    # optionally save similarity matrix
    if SAVE_RS_VALUES and rulesetfile and save_path:
        rr = pd.read_csv(rulesetfile, names=["Rule", "Covering", "Error"])
        df = pd.DataFrame(sim_matrix, index=rule_ids, columns=rule_ids)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_excel(f"{save_path}rulesimilarity.xlsx")

    return sim_matrix

