## CONFIDERAI+: CONFIDERAI extension to mutually exclusive rules

This code allows to perform conformal prediction for rule-based binary classifiers. It extends our previous score function, CONFIDERAI [1], to also include non-overlapping, mutually exclusive, rules, thus being applicable for Decision-Tree-like classifiers.

# How to use

The usage of this code on custom data is very simple and intuitive.

1) Set experiment configurations in `config.yaml`; in particular, set the `experiment_setup` to `synthetic` and `generate_data` to `True` to test the approaches on user-defined synthetic Gaussian data, as per the configs under `synthetic_data`. To work with real datasets, set the `experiment_setup` to `benchmark`: in this case, the code takes separated train, test, and calibration sets, first preprocessed via feature standardization.

2) Launch the experiment by running `test.sh`, after setting:
    - `datasetname` (scenario names if in `synthetic` setup, or set the same dataset folder name if in `benchmark` mode)
    - `relevance`: boolean to express whether to use relevance term in CONFIDERAI+/Risk Averse CONFIDERAI scores
    - `similarity`: whether to use rule similarity term in CONFIDERAI+/Risk Averse CONFIDERAI scores or not.

# References

[1] Narteni, S., Carlevaro, A., Dabbene, F., Muselli, M., & Mongelli, M. (2025). A novel score function for conformal prediction in rule-based binary classification. Pattern Recognition, 112219.
 