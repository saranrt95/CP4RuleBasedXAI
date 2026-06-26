
## CP4RULEBASEDXAI
This code allows to perform conformal prediction for rule-based binary classifiers. It extends our previous score function, CONFIDERAI [1], to also include non-overlapping, mutually exclusive, rules, being applicable for Decision-Tree-like sklearn-compatible classifiers.

[CONFIDERAI_comparison_concept.pdf](https://github.com/user-attachments/files/29377189/CONFIDERAI_comparison_concept.pdf)

The code implements two novel scores:

1) Risk-Averse CONFIDERAI
2) Risk-Tolerant CONFIDERAI 

and compares the results with more traditional score functions (LAC, MARGIN, and K-Nearest Neighbors).

# How to use

The usage of this code on custom data is very simple and intuitive.

1) Set experiment configurations in `config.yaml`; in particular, set the `experiment_setup` to `synthetic` and `generate_data` to `True` to test the approaches on user-defined synthetic Gaussian data, as per the configs under `synthetic_data`. To work with real datasets, set the `experiment_setup` to `benchmark`: in this case, the code takes separated train, test, and calibration sets, first preprocessed via feature standardization.

2) Launch the experiment by running `test.sh`, after setting:
    - `datasetname` (scenario names if in `synthetic` setup, or set the same dataset folder name if in `benchmark` mode)

# References

[1] Narteni, S., Carlevaro, A., Dabbene, F., Muselli, M., & Mongelli, M. (2025). A novel score function for conformal prediction in rule-based binary classification. Pattern Recognition, 112219.

# Citation
This code can be used to replicate the experiments of our paper _S. Narteni, A. Carlevaro, F. Dabbene, M. Mongelli "Rule-based Conformal Prediction for Risk-aware Decision Making" (2026)_, currently under submission at IEEE Transactions on Reliability journal.

 
