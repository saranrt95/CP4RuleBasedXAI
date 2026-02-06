import pandas as pd
import numpy as np
import re
from collections import defaultdict


def find_all(a_str, sub):
    """Find all start indices of substring `sub` in string `a_str`."""
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)

def flatten(lst):
    """Flatten a nested list."""
    return [item for sublist in lst for item in sublist]


operators = ['<', '>']

def get_condition_operators(cond, operators):
    """Extract comparison operators (<, <=, >, >=) from a condition string."""
    ops = []
    for opp in operators:
        idxlist = list(find_all(cond, opp))
        if len(idxlist) != 0:
            for opindex in idxlist:
                if cond[opindex + 1] == "=":
                    ops.append(cond[opindex:opindex + 2])
                else:
                    ops.append(cond[opindex])
    return ops


def get_dataframe_from_dict(parsedrule, ruleidlist):
    """Convert parsed rule dictionary to a DataFrame (safe version)."""
    rows = []
    rule_counter = 0

    for output in parsedrule:
        for cond in parsedrule[output]:
            if rule_counter >= len(ruleidlist):
                # if ruleidlist too short, assign last known or fallback ID
                rule_id = ruleidlist[-1] if len(ruleidlist) > 0 else 0
            else:
                rule_id = ruleidlist[rule_counter]
            rule_counter += 1

            # safeguard in case cond is incomplete
            if len(cond) < 3:
                continue

            rows.append({
                "Output Class": output,
                "Rule ID": rule_id,
                "Lower": cond[0],
                "Feature": cond[1],
                "Upper": cond[2]
            })

    ruledf = pd.DataFrame(rows, columns=["Output Class", "Rule ID", "Lower", "Feature", "Upper"])
    return ruledf


def rule_elements_extraction(ruleconditions, ruleidlist):
    """Extract feature, thresholds, and operators from each rule condition."""
    parsedrule = {}
    for output in ruleconditions:
        conditions = ruleconditions[output]
        cond_elements = []

        for cond in conditions:
            cond_elementslist = []
            ops = get_condition_operators(cond, operators)

            if len(ops) == 1:
                cond_elementslist = cond.split(ops[0])
                cond_elementslist = [e.strip() for e in cond_elementslist]
                if ops[0] in ['<=', '<']:
                    cond_elementslist = [""] + cond_elementslist + [""]
                elif ops[0] in ['>=', '>']:
                    cond_elementslist = [cond_elementslist[1], cond_elementslist[0], ""]
            elif len(ops) == 2:
                cond_elementslist = re.split(rf"{ops[0]}|{ops[1]}", cond)
                cond_elementslist = [e.strip() for e in cond_elementslist]
            else:
                continue

            if len(cond_elementslist) == 3:
                cond_elements.append(cond_elementslist)

        parsedrule[output] = cond_elements

    parsedruledf = get_dataframe_from_dict(parsedrule, ruleidlist)
    return parsedruledf


def impute_missing_thresholds(datafile, parsedruledf):
    """Replace missing lower/upper bounds with feature min/max."""
    #data = pd.read_excel(datafile)
    data = datafile
    for idx in range(len(parsedruledf)):
        feat = parsedruledf.at[idx, "Feature"]
        if parsedruledf.at[idx, "Lower"] == "":
            parsedruledf.at[idx, "Lower"] = data[feat].min()
        else:
            parsedruledf.at[idx, "Lower"] = float(parsedruledf.at[idx, "Lower"])
        if parsedruledf.at[idx, "Upper"] == "":
            parsedruledf.at[idx, "Upper"] = data[feat].max()
        else:
            parsedruledf.at[idx, "Upper"] = float(parsedruledf.at[idx, "Upper"])
    return parsedruledf


def fill_missing_features(datafile, ruleinfo, featurelabels, nfeatures):
    """Ensure each rule includes all features (impute missing ones)."""
    #data = pd.read_excel(datafile)
    data = datafile
    ruleid = list(set(list(ruleinfo["Rule ID"].values)))
    completerules = []

    for r in ruleid:
        thisruleinfo = ruleinfo[ruleinfo["Rule ID"] == r]
        thisruleinfo = thisruleinfo.groupby("Feature").agg({
            "Output Class": "first",
            "Rule ID": "first",
            "Lower": "max",
            "Upper": "min"
        }).reset_index()

        clslabel = list(set(thisruleinfo["Output Class"].values))[0]

        if len(thisruleinfo) != nfeatures:
            featureinrule = list(thisruleinfo["Feature"].values)
            featuremissing = [f for f in featurelabels if f not in featureinrule]
            lower_missing = [data[f].min() for f in featuremissing]
            upper_missing = [data[f].max() for f in featuremissing]
            tmp = pd.DataFrame({
                "Output Class": clslabel,
                "Rule ID": r,
                "Lower": lower_missing,
                "Feature": featuremissing,
                "Upper": upper_missing
            })
            completerules.append(pd.concat([thisruleinfo, tmp], ignore_index=True))
        else:
            completerules.append(thisruleinfo)

    completeruledf = pd.concat(completerules, ignore_index=True)
    completeruledf = completeruledf.sort_values(by=["Rule ID"], ascending=True, ignore_index=True)
    return completeruledf



def _parse_rule(rule_text, rule_id, outputlabel):
    # Extract output class
    match_class = re.search(rf'THEN\s+{outputlabel}\s+in\s+(\d+)', rule_text)
    output_class = int(match_class.group(1)) if match_class else None

    # Remove the RULE prefix and THEN suffix
    # 1) Remove "RULE n: IF"
    rule_body = re.sub(r'^RULE\s+\d+:\s+IF\s+', '', rule_text, flags=re.IGNORECASE)
    # 2) Remove THEN ... part
    rule_body = re.sub(rf'\s+THEN\s+{outputlabel}\s+in\s+\d+', '', rule_body, flags=re.IGNORECASE)

    # Now split on AND
    conditions = [c.strip() for c in re.split(r'\s+AND\s+', rule_body, flags=re.IGNORECASE)]
    #print(conditions)
    parsed_rows = []
    for cond in conditions:
        # Case 1: two-sided bounds: value1 < feature <= value2
        m = re.match(r'(-?\d+\.?\d*)\s*(<|<=)\s*([A-Za-z_][A-Za-z0-9_]*)\s*(<|<=|>|>=)\s*(-?\d+\.?\d*)', cond)
        #print("m: ", m)
        if m:
            val1, op1, feature, op2, val2 = m.groups()
            val1 = float(val1)
            val2 = float(val2)
            lower = val1 if op1 in ('<', '<=') else ""
            upper = val2 if op2 in ('<', '<=') else ""
            parsed_rows.append({
                'Output Class': output_class,
                'Rule ID': rule_id,
                'Feature': feature,
                'Lower': lower,
                'Upper': upper
            })
            continue

        # Case 2: single comparison: feature op value
        m2 = re.match(r'([A-Za-z_][A-Za-z0-9_]*)\s*(<|<=|>|>=)\s*(-?\d+\.?\d*)', cond)
        #print("m2: ", m2)
        if m2:
            feature, op, val = m2.groups()
            val = float(val)
            lower = val if op in ('>', '>=') else ""
            upper = val if op in ('<', '<=') else ""
            parsed_rows.append({
                'Output Class': output_class,
                'Rule ID': rule_id,
                'Feature': feature,
                'Lower': lower,
                'Upper': upper
            })
            continue

    return parsed_rows

def parse_rule(rule_text, rule_id, outputlabel):
    # Extract final class
    match_class = re.search(rf'THEN\s+{outputlabel}\s+in\s+(\d+)', rule_text)
    output_class = int(match_class.group(1)) if match_class else None

    # Remove prefix and suffix
    rule_body = re.sub(r'^RULE\s+\d+:\s+IF\s+', '', rule_text, flags=re.I)
    rule_body = re.sub(rf',?\s*THEN\s+{outputlabel}\s+in\s+\d+.*$', '', rule_body, flags=re.I)

    # Split into atomic constraints
    conditions = [c.strip() for c in re.split(r'\s+AND\s+', rule_body)]

    parsed = []

    for cond in conditions:

        # Case A: X OP value
        m1 = re.match(r'([A-Za-z_]\w*)\s*(<=|<|>=|>)\s*(-?\d+\.?\d*)', cond)
        # Case B: value OP X
        m2 = re.match(r'(-?\d+\.?\d*)\s*(<=|<|>=|>)\s*([A-Za-z_]\w*)', cond)

        if m1:
            feature, op, val = m1.groups()
            val = float(val)
        elif m2:
            val, op, feature = m2.groups()
            val = float(val)
            # Reverse operator
            op = {'<':'>','<=':'>=','>':'<','>=':'<='}[op]
        else:
            continue

        # Convert operator into lower/upper bound semantics
        if op in ('>', '>='):
            parsed.append({"Output Class": output_class,
                "Rule ID": rule_id,
                "Feature": feature,
                "Lower": val,
                "Upper": ""
            })
        elif op in ('<', '<='):
            parsed.append({"Output Class": output_class,
                "Rule ID": rule_id,
                "Feature": feature,
                "Lower": "",
                "Upper": val
            })

    return parsed

def merge_feature_constraints(df_feat, data):
    """Given all rows for a single (rule, feature), intersect bounds."""

    feature = df_feat["Feature"].iloc[0]

    lowers = [v for v in df_feat["Lower"] if v != ""]
    uppers = [v for v in df_feat["Upper"] if v != ""]

    lower = max(lowers) if lowers else data[feature].min()
    upper = min(uppers) if uppers else data[feature].max()

    return dict(
        Feature=feature,
        Lower=lower,
        Upper=upper
    )

def clean_ruleset_file(rulesetfile, datafile, featurelabels, outputlabel):

    rules = list(pd.read_csv(rulesetfile, header=None)[0])
    data = datafile

    all_rows = []
    rule_id = 0

    for rule in rules:
        rule_id += 1
        parsed = parse_rule(rule, rule_id, outputlabel)
        all_rows.extend(parsed)

    df = pd.DataFrame(all_rows)

    # Ensure numeric
    df["Rule ID"] = df["Rule ID"].astype(int)
    df["Output Class"] = df["Output Class"].astype(int)

    final_rows = []

    for rid in df["Rule ID"].unique():

        rule_rows = df[df["Rule ID"] == rid]
        output_class = rule_rows["Output Class"].iloc[0]

        features_present = rule_rows["Feature"].unique()
        missing = [f for f in featurelabels if f not in features_present]

        # Merge existing feature constraints
        merged = []
        for feat in features_present:
            df_feat = rule_rows[rule_rows["Feature"] == feat]
            merged.append(merge_feature_constraints(df_feat, data))

        # Add missing features with full range
        for feat in missing:
            merged.append(dict(
                Feature=feat,
                Lower=data[feat].min(),
                Upper=data[feat].max()
            ))

        # Add output class and rule id
        for row in merged:
            row["Rule ID"] = rid
            row["Output Class"] = output_class
            final_rows.append(row)

    final_df = pd.DataFrame(final_rows)
    final_df = final_df.sort_values(["Rule ID", "Feature"]).reset_index(drop=True)
    return final_df

def _clean_ruleset_file(rulesetfile, datafile, featurelabels, nfeatures, outputlabel):
    """Main function: parse, clean, and complete rule set into structured DataFrame."""
    ruledata = pd.read_csv(rulesetfile, header=None)
    rules = list(ruledata[0])
    rc = 0

    parsed_rules = []
    for rule in rules:
        rc+=1
        parsed_rule = parse_rule(rule, rc, outputlabel)
        parsed_rules.append(parsed_rule)
      
    flat_parsed_list = [item for sublist in parsed_rules for item in sublist]
    parsedruledf = pd.DataFrame(flat_parsed_list)
    # impute missing bounds
    filledruledf = impute_missing_thresholds(datafile, parsedruledf)

    # fill missing features
    filledruledf = filledruledf.astype({'Rule ID': 'int64', 'Output Class': 'int64'})
    completeruledf = fill_missing_features(datafile, filledruledf, featurelabels, nfeatures)
    completeruledf = completeruledf.astype({'Rule ID': 'int64', 'Output Class': 'int64'})

    return completeruledf



