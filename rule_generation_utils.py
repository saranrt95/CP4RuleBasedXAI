import numpy as np
from sklearn import tree
from sklearn.tree import _tree
import numpy as np
import re
import pandas as pd


def get_rule_points(lowerX1, upperX1, lowerX2, upperX2, numPoints):
    w = abs(upperX1 - lowerX1)
    h = abs(upperX2 - lowerX2)
    
    xRandom = lowerX1 + w * np.random.rand(numPoints)
    yRandom = lowerX2 + h * np.random.rand(numPoints)
    
    X_r = np.column_stack((xRandom, yRandom))
    
    return X_r, w, h

def write_and_save_ruleset(rule_thresholds, save_path = "./"):
    open(save_path, 'w')
    for count, (l1,u1,l2,u2,cls) in enumerate(rule_thresholds):
        with open(save_path, 'a') as ff:
            ff.write(f"RULE {count+1}: IF {l1} < X1 <= {u1} AND {l2} < X2 <= {u2} THEN output in {cls},COVERING: {np.random.uniform(0, 1)},ERROR: {np.random.uniform(0, 0.1)}\n")
    ff.close()

##### custom data utils #####

def MixGauss(means, sigmas, n):
    means = np.asarray(means)
    sigmas = np.asarray(sigmas)

    d, p = means.shape
    X = []
    Y = []

    for i in range(p):
        m = means[:, i]
        S = sigmas[i]
        # Generate n samples for this Gaussian
        Xi = S * np.random.randn(n, d) + m
        Yi = np.full(n, i)
        X.append(Xi)
        Y.append(Yi)

    X = np.vstack(X)
    Y = np.concatenate(Y)
    return X, Y

def generate_data(N_points, p_A, p_O, mu1, mu2, sigma1, sigma2, random_state=None):
    #p_B = 1 - p_A
    if random_state is not None:
        np.random.seed(random_state)
    X_red, Y_red = [], []
    X_blue, Y_blue = [], []

    for _ in range(N_points):
        if np.random.rand() < p_A:
            # Class A (red)
            if np.random.rand() < p_O:  # Outlier
                x_i, y_i = MixGauss(mu1, sigma1, 1)
                y_i[y_i == 0] = 0
            else:
                x_i, y_i = MixGauss(mu1, sigma1, 1)
                y_i[y_i == 0] = 1
            X_red.append(x_i[0])
            Y_red.append(y_i[0])
        else:
            # Class B (blue)
            if np.random.rand() < p_O:  # Outlier
                x_i, y_i = MixGauss(mu2, sigma2, 1)
                y_i[y_i == 0] = 1
            else:
                x_i, y_i = MixGauss(mu2, sigma2, 1)
                y_i[y_i == 0] = 0
            X_blue.append(x_i[0])
            Y_blue.append(y_i[0])

    X = np.vstack(X_red + X_blue)
    Y = np.concatenate([Y_red, Y_blue])

    return X, Y

#### dt training utils ####

# this function extracts the rules from a Decision Tree model and writes them as IF-THEN sentences

def ExtractRulesFromDT(tree, feature_names, class_names, target_label):
    tree_ = tree.tree_

    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []

    def recurse(node, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], path + [f"{name} <= {np.round(threshold, 3)}"])
            recurse(tree_.children_right[node], path + [f"{name} > {np.round(threshold, 3)}"])
        else:
            value = tree_.value[node]
            n_samples = tree_.n_node_samples[node]
            paths.append((path, value, n_samples))

    recurse(0, [])

    # Extract relevant info
    leaves_info = []
    for path, value, n_samples in paths:
        class_idx = int(np.argmax(value))
        class_label = class_names[class_idx] if class_names else str(class_idx)
        leaves_info.append({
            "path": path,
            "class_idx": class_idx,
            "class_label": class_label,
            "samples": n_samples,
            "value": value
        })

    # Sort by (class_idx, -samples)
    leaves_info = sorted(leaves_info, key=lambda x: (x["class_idx"], -x["samples"]))

    # Build rule strings and find first rule predicting the 2nd class
    rules = []
    changeclsidx = None
    second_class_idx = 1 if len(class_names) > 1 else None

    for i, leaf in enumerate(leaves_info, start=1):
        rule = f"RULE {i}: IF " + " AND ".join(leaf["path"])
        rule += f" THEN {target_label} in {leaf['class_label']}"
        rules.append(rule)

        if changeclsidx is None and leaf["class_idx"] == second_class_idx:
            changeclsidx = i

    return rules, changeclsidx



def computeRuleMetrics(rule,X_train,y_train):
    conditions = rule.split("IF ")[1].split(" AND ")
    output_rule = np.dtype(y_train).type(conditions[-1].split(" THEN ")[1].split(" in ")[1])
    #print(output_rule)
    conditions[-1] = conditions[-1].split(" THEN ")[0]
    #print(conditions)
    true_negatives = (y_train[X_train.query("not(" + " and ".join(conditions) + ")").index]!=output_rule).sum()
    false_positives = (y_train[X_train.query(" and ".join(conditions)).index]!=output_rule).sum()
    true_positives = (y_train[X_train.query(" and ".join(conditions)).index]==output_rule).sum()
    false_negatives = (y_train[X_train.query("not(" + " and ".join(conditions) + ")").index]==output_rule).sum()
    
    if true_positives+false_negatives == 0:
        covering = 0
    else:
        covering = true_positives / (true_positives + false_negatives)
    if true_negatives + false_positives == 0:
        error = 0      
    else:

        error = false_positives / (true_negatives + false_positives)

    return covering, error


def extract_rule_thresholds_from_df(file_path, df):
    """
    Extract thresholds for each rule in the format:
    [lower_X1, upper_X1, lower_X2, upper_X2]
    """
    # Regex supports >, >=, <, <=
    condition_pattern = re.compile(r'(X\d+)\s*(<=|>=|<|>)\s*([-+]?\d*\.?\d+)')
    
    rule_thresholds = []

    # Dataset min/max
    data_min = df.min().to_dict()
    data_max = df.max().to_dict()

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith("RULE"):
                continue

            # Start with dataset-wide bounds
            lower = {var: data_min[var] for var in data_min}
            upper = {var: data_max[var] for var in data_max}

            # Extract all feature conditions
            matches = condition_pattern.findall(line)

            for var, op, value in matches:
                value = float(value)

                if op in ('>', '>='):
                    # increase lower bound
                    lower[var] = max(lower[var], value)
                elif op in ('<', '<='):
                    # decrease upper bound
                    upper[var] = min(upper[var], value)

            # Order is [lower_X1, upper_X1, lower_X2, upper_X2]
            rule_thresholds.append([
                lower['X1'], upper['X1'], 
                lower['X2'], upper['X2']
            ])

    return rule_thresholds

def extract_rule_thresholds_from_df_multid(file_path, df):
    """
    Extract thresholds for each rule in the format:
    [l1,u1,l2,u2,...,ld,ud]

    Returns
    -------
    rule_thresholds : list of lists
    feature_order   : list of feature names
    """

    condition_pattern = re.compile(
        r'(X\d+)\s*(<=|>=|<|>)\s*([-+]?\d*\.?\d+)'
    )

    # Feature order inferred from dataframe
    feature_order = list(df.columns)

    # Dataset-wide bounds
    data_min = df.min().to_dict()
    data_max = df.max().to_dict()

    rule_thresholds = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith("RULE"):
                continue

            # initialize bounds
            lower = {feat: data_min[feat] for feat in feature_order}
            upper = {feat: data_max[feat] for feat in feature_order}

            matches = condition_pattern.findall(line)

            for var, op, value in matches:
                value = float(value)

                if op in ('>', '>='):
                    lower[var] = max(lower[var], value)
                elif op in ('<', '<='):
                    upper[var] = min(upper[var], value)

            # flatten in consistent order
            rule_limits = []
            for feat in feature_order:
                rule_limits.extend([lower[feat], upper[feat]])

            rule_thresholds.append(rule_limits)
    return rule_thresholds, feature_order

def ComputeRelevances(rulesetfile):
	
	""" For each rule of the ruleset, computes its relevance as covering*(1-error); 
		Input: 
			- rulesetfile: path to file with the ruleset (IF-THEN csv format), reporting covering and error metrics
		Output:
			- relevances: numpy array with the obtained relevance values. 

	"""
	errorlist=[]
	coveringlist=[]
	with open(rulesetfile,"r") as infile:
		rules = infile.readlines()
		#print(rules)
		for rule in rules:
			#print(rule)
			rule = rule.replace("\n","")
			c = rule.split(",")[1]
			covering_value = float(c[11:])
			#print(covering_value)
			coveringlist.append(covering_value)
			e = rule.split(",")[2]
			#print(e[8:])
			error_value = float(e[8:])
			#print(error_value)
			errorlist.append(error_value)

	relevances = [c*(1-e) for c,e in list(zip(coveringlist,errorlist))]
	relevances = np.array(relevances)
	return relevances


def ReformattingRuleset(rules, output_label):
    #print(output_label)
    # adjust columns values
    for i in range(len(rules)+1):   
        rules["Rule"] = rules["Rule"].apply(lambda x: x.replace("RULE {}: ".format(i),""))
    rules["Rule"] = rules["Rule"].apply(lambda x: x.replace("AND","and"))
    rules["Rule"] = rules["Rule"].apply(lambda x: x.replace("{",""))
    rules["Rule"] = rules["Rule"].apply(lambda x: x.replace("}",""))
    rules["Rule"] = rules["Rule"].apply(lambda x: x.replace(f"{output_label} in ",f"{output_label} = "))
    rules["Covering"] = rules["Covering"].apply(lambda x: x.replace("COVERING: ",""))
    rules["Error"] = rules["Error"].apply(lambda x: x.replace("ERROR: ",""))
    rules['Output'] = rules['Rule'].str.extract(rf'{output_label} = (\d+)', expand=False).astype(int)
    rules["Rule"] = rules["Rule"].apply(lambda x: x.replace("IF ",""))
    rules["Rule"] = rules["Rule"].apply(lambda x: x.replace(x[x.find("THEN"):],""))
    
    return rules
def check_condition(row, condition_part):
    # Check if a single condition part is satisfied
    
    #parts = [part.strip('()') for part in condition_part.split()]
    #parts = [part for part in condition_part.split()]
    # Use regular expressions to properly parse the condition
    parts = re.split(r'\s*(==|<=|>=|<|>|!=)\s*', condition_part)
    
    #print("parts: ", parts)
    if len(parts) == 3:
        column, op, value = parts
        return eval(f"{row[column]} {op} {value}")
    # handle the case of a 2-thresholds conditions of the kind: a < Column <= b
    elif len(parts) == 5:
        val1,op1,column,op2,val2 = parts
        # Use the original condition from the rule
        return eval(f"{val1} {op1} {row[column]} {op2} {val2}")        
    else:
        raise ValueError("Bad condition formatting!")
    
def evaluate_rule_conditions(row, condition_part):

    # Checks if any of the conditions in the rule are satisfied
    if all(check_condition(row, part) for part in condition_part.split(" and ")):
        return True  # Return True if all conditions in the rule are satisfied
    
    return False  # Return False if any of the conditions in the rule is not satisfied


def GetSatisfiedRules(data, rulesetfile, output_label):
    rules = pd.read_csv(rulesetfile, header=None, names=["Rule", "Covering", "Error"])
    rules = ReformattingRuleset(rules, output_label)
    #print(rules["Rule"])
    idx_rules = 0
    satisfiedMat = np.zeros((len(data),len(rules)))
    for i, rule in rules.iterrows():
        idx_data = 0
        tuned_antecedent = rule['Rule'].strip()
        #print("rule: ", tuned_antecedent)
        for j, row in data.iterrows():
            
            # check if the point row satifies rule 
            if evaluate_rule_conditions(row, tuned_antecedent):
                # rule is satisfied           
                satisfiedMat[idx_data,idx_rules] = 1
                #print("satisfied")
            else:
                satisfied = False
            idx_data+=1
            #print("not satisfied")
        idx_rules+=1
    # given the matrix of satisfied rules, get the corresponding rule indexes
    #verified_rules_indexes = [list(np.where(row == 1)[0] + 1) for row in satisfiedMat] 
    return satisfiedMat#verified_rules_indexes