import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from rule_generation_utils import *

def train_decision_tree(data_tr, output = "output", max_n_rules = 10, ccp_alpha = 0, save_model = True, model_path = None):
    y_train=data_tr[output]
    data_features = data_tr.drop([output],axis=1)
    #train sklearn model on the proper training set 
    model = DecisionTreeClassifier(random_state=0, max_leaf_nodes=max_n_rules, ccp_alpha=ccp_alpha)
    model = model.fit(data_features,y_train)
    if save_model and model_path!= None:
        joblib.dump(model, model_path)
    return model 

def get_decision_tree_predictions(trained_model, data_ts, output = "output"):
    data_features = data_ts.drop([output],axis=1)
    y_pred_ts = trained_model.predict(data_features)
    return y_pred_ts

def extract_and_save_rules(model, res_path, rulesetfile_full, data_tr, covering_threshold = 0.1, output = "output", save = False):
    y_train=data_tr[output]
    data_features = data_tr.drop([output],axis=1)
    #data_tr.drop(["output"],axis=1,inplace=True)
    # rule extraction
    rules, changeclsidx = ExtractRulesFromDT(data_features, y_train, model, list(model.feature_names_in_), [0,1], output, covering_threshold=covering_threshold)
    if save:
        open(f"{res_path}/{rulesetfile_full}","w")
        for rule in rules:
            print(rule)
            #covering, error = computeRuleMetrics(rule,data_features,y_train)
            #if covering >= covering_threshold:
            with open(f"{res_path}/{rulesetfile_full}","a") as rulefile:
                #print("saving to file")
                rulefile.write(str(rule)+"\n")#+f", COVERING: {covering}, ERROR: {error}\n")
    # Rule thresholds
    rule_limits,_ = extract_rule_thresholds_from_df_multid(f"{res_path}/{rulesetfile_full}", data_features)#)
    rule_limits = np.array(rule_limits)

    #print(rule_limits.shape)
    nrules = rule_limits.shape[0]

    relevance = ComputeRelevances(f"{res_path}/{rulesetfile_full}")#np.zeros(nrules)

    return rule_limits, changeclsidx, nrules, relevance 

