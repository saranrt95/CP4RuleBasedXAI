
import yaml
import os
import sys


import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report
import joblib
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names*")

from data_utils import *
from model_utils import *

from rule_generation_utils import *
from rulesetparsing import *
from geometric_rule_similarity import *

from scores.conformal_utils import *
from scores.risk_tolerant_confiderai import *
from scores.risk_averse_confiderai import * 
from scores.lac import *
from scores.margin import *
from scores.knn_utils import *

from plot_utils import *




def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()



# command line variables
if len(sys.argv)!=5:
    print(f"Usage: {sys.argv[0]} <dataset> <use_relevance (true|false)> <use_similarity (true|false)>")
    sys.exit(0)
datasetname = sys.argv[1]
USE_RELEV = sys.argv[2].lower() == "true"
USE_SIM = sys.argv[3].lower() == "true"

print("DATASET: ", datasetname)


# CP settings
epsilonrange = config["cp"]["epsilon_range"]
n_eps = len(epsilonrange)
DISTANCE = config["cp"]["risk_tolerant"]["distance"]
P = config["cp"]["risk_tolerant"]["p"]
NORM_FN = sys.argv[4] #config["cp"]["risk_tolerant"]["normalization"]
AGGREGATION = config["cp"]["risk_tolerant"]["aggregation"]
BETA = config["cp"]["risk_tolerant"]["beta"]
K = config["cp"]["risk_tolerant"]["norm_fun"]["sigmoid"]["k"]
SHIFT = config["cp"]["risk_tolerant"]["norm_fun"]["sigmoid"]["shift"]
# paths for folders and files
res_dir = config['output']["res_dir"]
res_path = f"{res_dir}/{datasetname}_Rel{USE_RELEV}_Sim{USE_SIM}_{NORM_FN}/"
os.makedirs(res_path, exist_ok=True)

save_plots_flag = config['output']["save_plots"]

if save_plots_flag:
    plots_dir = os.path.join(res_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)



rulesetfile = config['model']["ruleset_file"]
rulesim_path = ""

# synthetic dataset settings
if config['experiment_type']=='synthetic':
    n_clusters_per_class = config["synthetic_dataset"]["n_clusters_per_class"]
    N_points = config["synthetic_dataset"]["n_points"]
    n_features = config['synthetic_dataset']['n_features']
    # Scenario-specific parameters
    scenario_cfg = config["synthetic_dataset"]["scenarios"][datasetname]
    centers = scenario_cfg["centers"]
    sigmas = scenario_cfg["sigmas"]
    outfile = os.path.join(res_path, "synthetic_dataset.xlsx")


with open(f"{res_path}/metrics.csv","w") as ff:
    ff.write("score,epsilon,avgErr,stdErr,avgSingleton,stdSingleton,avgDouble,stdDouble,avgEmpty,stdEmpty,avgSize,stdSize,errSingleton,stdErrSingleton\n")

## MAIN ###

# generate synthetic Gaussian data according to the scenario
if config['experiment_type']=='synthetic':

    if config['synthetic_dataset']['generate_data']:
        outputlabel = config['synthetic_dataset']['outputlabel']
        cls0label = config['synthetic_dataset']['cls0label']
        cls1label = config['synthetic_dataset']['cls1label']
        X, Y = make_blobs(n_samples=N_points, centers=centers, n_features = n_features, cluster_std=sigmas, shuffle=False, random_state=42)
        # create binary classes, assuming the first half clusters are for class 1, otherwise for class 0
        Y = np.where(np.isin(Y, list(range(n_clusters_per_class))), cls1label, cls0label)
        plot_classes(X, Y, save_plots_flag=save_plots_flag, res_path=res_path)
        data_combined = np.hstack([X, Y.reshape(-1, 1)])
        df = pd.DataFrame(data_combined, columns=[f"X{i+1}" for i in range(n_features)]+[outputlabel])
        if config["synthetic_dataset"]["save_data"]:
            df.to_excel(outfile, index=False)
    else:
        df = pd.read_excel(outfile)

    data_tr, data_cal, data_ts = split_data(df)
    featurelabels = list(data_tr.columns)[:-1]
    outputlabel = list(data_tr.columns)[-1]
    cls0label, cls1label = list(set(data_tr[outputlabel]))
# benchmark real datasets
elif config['experiment_type']=='benchmark':
    data_tr = pd.read_excel(f"{config["dataset"]["data_path"]}/{datasetname}/proper.xlsx")
    data_cal = pd.read_excel(f"{config["dataset"]["data_path"]}/{datasetname}/calibration.xlsx")
    data_ts = pd.read_excel(f"{config["dataset"]["data_path"]}/{datasetname}/test.xlsx")
    featurelabels = list(data_tr.columns)[:-1]
    n_features = len(featurelabels)
    outputlabel = list(data_tr.columns)[-1]
    cls0label, cls1label = list(set(data_tr[outputlabel]))

print(f"# train: {len(data_tr)}, # calibration: {len(data_cal)}, # test: {len(data_ts)}")

# re-arrange to numpy and separate X and Y
Xts = data_ts.loc[:, featurelabels].to_numpy()
Yts = data_ts.loc[:, outputlabel].to_numpy()
Xcal = data_cal.loc[:, featurelabels].to_numpy()
Ycal = data_cal.loc[:, outputlabel].to_numpy()
n_c = Xcal.shape[0]

if config['model']["train_model"]:
    max_n_rules = config["model"]["max_n_rules"]
    ccp_alpha = config["model"]["ccp_alpha"]
    min_samples_thr = config["model"]["min_samples_thr"]
    model = train_decision_tree(data_tr, output=outputlabel, min_samples_rule_thr=min_samples_thr, max_n_rules = max_n_rules, ccp_alpha=ccp_alpha, save_model = True, model_path = f"{res_path}/{config['model']['model_name']}.sav") # train_IREP(data_tr, output=outputlabel, save_model = True, model_path = f"{res_path}/{config['model']['model_name']}.sav")#
else:
    model = joblib.load(f"{res_path}/{config['model_name']}.sav")


y_pred_ts = get_decision_tree_predictions(model, data_ts, output=outputlabel)
print(f"Performance of the DT on test data (max_leaf_nodes = {model.max_leaf_nodes})")
print(classification_report(data_ts[outputlabel], y_pred_ts))
y_pred_cal = get_decision_tree_predictions(model, data_cal, output=outputlabel)


### get misclassified samples ###
wrong_0 = Xcal[(Ycal == cls1label) & (y_pred_cal == cls0label)]
wrong_1 = Xcal[(Ycal == cls0label) & (y_pred_cal == cls1label)]
wrong_0_ts = Xts[(Yts == cls1label) & (y_pred_ts == cls0label)]
wrong_1_ts = Xts[(Yts == cls0label) & (y_pred_ts == cls1label)]

# extract rules from DT and compute rule relevances

rule_limits, changeclsidx, nrules, relevance = extract_and_save_rules(model, res_path, rulesetfile, data_tr, covering_threshold = 0.0, output=outputlabel, save = config['model']['train_model'])


# parse the ruleset to extract each condition and fill each rule with missing thresholds
parsedruleset = clean_ruleset_file(f"{res_path}/{rulesetfile}", data_tr, featurelabels, outputlabel)
parsedruleset.Feature = parsedruleset.Feature.astype("category")
parsedruleset.Feature = parsedruleset.Feature.cat.set_categories(featurelabels)
parsedruleset = parsedruleset.sort_values(["Rule ID", "Feature"])
rulesim = GeneralizedIoU(parsedruleset, rulesetfile=f"{res_path}/{rulesetfile}", SAVE_RS_VALUES=True, save_path=res_path+rulesim_path) 

###### RT-CONFIDERAI #########

print("RT-CONFIDERAI")

tau0cal, gamma0cal,simterm0cal = compute_confiderai_score(Xcal, rulesim, rule_limits, changeclsidx, cls0label, relevance,  distance = DISTANCE, p = P, norm_function = NORM_FN, aggregation = AGGREGATION, beta = BETA, k=K, shift=SHIFT, use_relevance=USE_RELEV, use_sim=USE_SIM)
tau1cal, gamma1cal,simterm1cal = compute_confiderai_score(Xcal, rulesim, rule_limits, changeclsidx, cls1label, relevance,  distance = DISTANCE, p = P, norm_function = NORM_FN, aggregation = AGGREGATION, beta = BETA, k=K, shift=SHIFT, use_relevance=USE_RELEV, use_sim=USE_SIM)
tau0ts, _,_ = compute_confiderai_score(Xts, rulesim, rule_limits, changeclsidx, cls0label, relevance,  distance = DISTANCE, p = P, norm_function = NORM_FN, aggregation = AGGREGATION, beta = BETA, k=K, shift=SHIFT, use_relevance=USE_RELEV, use_sim=USE_SIM)
tau1ts,_,_ = compute_confiderai_score(Xts, rulesim, rule_limits, changeclsidx, cls1label, relevance,  distance = DISTANCE, p = P, norm_function = NORM_FN, aggregation = AGGREGATION, beta = BETA, k=K, shift=SHIFT, use_relevance=USE_RELEV, use_sim=USE_SIM)

selectedscores_cal = np.where(Ycal == cls0label, tau0cal, tau1cal)

if n_features == 2:
    plot_score(Xts, tau0ts, tau1ts, rule_limits, changeclsidx, wrong_1_ts, wrong_0_ts, save_plots_flag = save_plots_flag, score_fn = "RT-CONFIDERAI", res_path = res_path)


avgErr, varErr, avgErr_singleton, varErr_singleton, empty, var_empty, singleton, var_singleton, double, var_double, avgSize, varSize = evaluate_conformal(Ycal, tau0cal, tau1cal, n_c, epsilonrange, Yts, tau0ts, tau1ts, res_path, "RT-CONFIDERAI")


plot_metrics(epsilonrange, avgErr, avgSize, "RT-CONFIDERAI", save_plots_flag = save_plots_flag, res_path = res_path, show = False)

plot_calibration_scores_distribution(selectedscores_cal, "RT-CONFIDERAI", epsilon_vals = [0.01, 0.05, 0.1, 0.2], save_plots_flag = save_plots_flag, res_path = res_path, show = False)
if n_features == 2:
    plot_prediction_regions(Xts, Ycal, tau0cal, tau1cal, tau0ts, tau1ts, rule_limits, changeclsidx, wrong_0_ts, wrong_1_ts, "RT-CONFIDERAI", selected_eps = 0.01, save_plots_flag = save_plots_flag, res_path = res_path, show = False)

### RISK-AVERSE CONFIDERAI ####
print("RA-CONFIDERAI")
tau0cal, gamma0cal,simterm0cal, _ = compute_dataset_score(Xcal, rulesim, rule_limits, changeclsidx, cls0label, relevance)
tau1cal, gamma1cal,simterm1cal, _ = compute_dataset_score(Xcal, rulesim, rule_limits, changeclsidx, cls1label, relevance)
tau0ts, _,_,_ = compute_dataset_score(Xts, rulesim, rule_limits, changeclsidx, cls0label, relevance)
tau1ts,_,_,_ = compute_dataset_score(Xts, rulesim, rule_limits, changeclsidx, cls1label, relevance)

selectedscores_cal = np.where(Ycal == cls0label, tau0cal, tau1cal)
if n_features == 2:
    plot_score(Xts, tau0ts, tau1ts, rule_limits, changeclsidx, wrong_1_ts, wrong_0_ts, save_plots_flag = save_plots_flag, score_fn = "RA-CONFIDERAI", res_path = res_path)


avgErr, varErr, avgErr_singleton, varErr_singleton, empty, var_empty, singleton, var_singleton, double, var_double, avgSize, varSize = evaluate_conformal(Ycal, tau0cal, tau1cal, n_c, epsilonrange, Yts, tau0ts, tau1ts, res_path, "RA-CONFIDERAI")


plot_metrics(epsilonrange, avgErr, avgSize, "RA-CONFIDERAI", save_plots_flag = save_plots_flag, res_path = res_path, show = False)

plot_calibration_scores_distribution(selectedscores_cal, "RA-CONFIDERAI", epsilon_vals = [0.01, 0.05, 0.1, 0.2], save_plots_flag = save_plots_flag, res_path = res_path, show = False)
if n_features == 2:
    plot_prediction_regions(Xts, Ycal, tau0cal, tau1cal, tau0ts, tau1ts, rule_limits, changeclsidx, wrong_0_ts, wrong_1_ts, "RA-CONFIDERAI", selected_eps = 0.01, save_plots_flag = save_plots_flag, res_path = res_path, show = False)



##### MARGIN ##########
print("MARGIN")
y_train = data_tr[outputlabel]
scores0_margin = margin_based_score(model, Xcal, y_train, cls0label, mode = "m-estimate")
scores1_margin = margin_based_score(model, Xcal, y_train, cls1label, mode = "m-estimate")
scores0ts_margin = margin_based_score(model, Xts, y_train, cls0label, mode = "m-estimate")
scores1ts_margin = margin_based_score(model, Xts, y_train, cls1label, mode = "m-estimate")

selectedscores_cal = np.where(Ycal == cls0label, scores0_margin, scores1_margin)
if n_features == 2:
    plot_score(Xts, scores0ts_margin, scores1ts_margin, rule_limits, changeclsidx, wrong_1_ts, wrong_0_ts, save_plots_flag = save_plots_flag, score_fn = "margin", res_path = res_path)

avgErr, varErr, avgErr_singleton, varErr_singleton, empty, var_empty, singleton, var_singleton, double, var_double, avgSize, varSize = evaluate_conformal(Ycal, scores0_margin, scores1_margin, n_c, epsilonrange, Yts, scores0ts_margin, scores1ts_margin, res_path, "margin")
    


plot_metrics(epsilonrange, avgErr, avgSize, "margin", save_plots_flag = save_plots_flag, res_path = res_path, show = False)

plot_calibration_scores_distribution(selectedscores_cal, "margin", epsilon_vals = [0.01, 0.05, 0.1, 0.2], save_plots_flag = save_plots_flag, res_path = res_path, show = False)
if n_features == 2:
    plot_prediction_regions(Xts, Ycal, scores0_margin, scores1_margin, scores0ts_margin, scores1ts_margin, rule_limits, changeclsidx, wrong_0_ts, wrong_1_ts, "margin", selected_eps = 0.01, save_plots_flag = save_plots_flag, res_path = res_path, show = False)

#### LAC ########
print("LAC")
scores0_lac = LAC_score(model, Xcal, cls0label)
scores1_lac = LAC_score(model, Xcal, cls1label)
scores0ts_lac = LAC_score(model, Xts, cls0label)
scores1ts_lac = LAC_score(model, Xts, cls1label)

selectedscores_cal = np.where(Ycal == cls0label, scores0_lac, scores1_lac)
if n_features == 2:
    plot_score(Xts, scores0ts_lac, scores1ts_lac, rule_limits, changeclsidx, wrong_1_ts, wrong_0_ts, save_plots_flag = save_plots_flag, score_fn = "LAC", res_path = res_path)

avgErr, varErr, avgErr_singleton, varErr_singleton, empty, var_empty, singleton, var_singleton, double, var_double, avgSize, varSize = evaluate_conformal(Ycal, scores0_lac, scores1_lac, n_c, epsilonrange, Yts, scores0ts_lac, scores1ts_lac, res_path, "LAC")
    

plot_metrics(epsilonrange, avgErr, avgSize, "LAC", save_plots_flag = save_plots_flag, res_path = res_path, show = False)

plot_calibration_scores_distribution(selectedscores_cal, "LAC", epsilon_vals = [0.01, 0.05, 0.1, 0.2], save_plots_flag = save_plots_flag, res_path = res_path, show = False)
if n_features == 2:
    plot_prediction_regions(Xts, Ycal, scores0_lac, scores1_lac, scores0ts_lac, scores1ts_lac, rule_limits, changeclsidx, wrong_0_ts, wrong_1_ts, "LAC", selected_eps = 0.01, save_plots_flag = save_plots_flag, res_path = res_path, show = False)

######### KNN ###########

print("KNN")
# output rule partitions and scores 
Xcal_rules, Ycal_rules, scores0_cal_knn, scores1_cal_knn = KNN_Score(Xcal, Ycal, f"{res_path}/{rulesetfile}", featurelabels, outputlabel, cls0label, cls1label, K=int(config["cp"]["knn"]["K"]),min_samples_rule=int(config["cp"]["knn"]["min_samples"]))
Xts_rules, Yts_rules, scores0_test_knn, scores1_test_knn = KNN_Score(Xts, Yts, f"{res_path}/{rulesetfile}", featurelabels, outputlabel, cls0label, cls1label, K=int(config["cp"]["knn"]["K"]),min_samples_rule=int(config["cp"]["knn"]["min_samples"]))

#selected_cal_rulewise = []
#for r in range(len(Xcal_rules)):
#    selectedscores_cal = np.where(Ycal_rules[r] == cls0label, scores0_cal_knn[r], scores1_cal_knn[r])
#    selected_cal_rulewise.append(selectedscores_cal)

#if n_features == 2:
#    plot_score(Xts, scores0_test_knn, scores1_test_knn, rule_limits, changeclsidx, wrong_1_ts, wrong_0_ts, save_plots_flag = save_plots_flag, score_fn = "knn", res_path = res_path)


avgErr, varErr, avgErr_singleton, varErr_singleton, empty, var_empty, singleton, var_singleton, double, var_double, avgSize, varSize = evaluate_conformal_knn(Ycal_rules, scores0_cal_knn, scores1_cal_knn, epsilonrange, Yts_rules, scores0_test_knn, scores1_test_knn, res_path, "knn")



plot_metrics(epsilonrange, avgErr, avgSize, "knn", save_plots_flag = save_plots_flag, res_path = res_path, show = False)

plot_calibration_scores_distribution(selectedscores_cal, "knn", epsilon_vals = [0.01, 0.05, 0.1, 0.2], save_plots_flag = save_plots_flag, res_path = res_path, show = False)
#if n_features == 2:
#    plot_prediction_regions(Xts, Ycal, scores0_cal_knn, scores1_cal_knn, scores0_test_knn, scores1_test_knn, rule_limits, changeclsidx, wrong_0_ts, wrong_1_ts, "knn", selected_eps = 0.01, save_plots_flag = save_plots_flag, res_path = res_path, show = False)
