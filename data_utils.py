import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(df):
    """ returns training, calibration and test data as pandas dataframes"""
    


    data_train_cal, data_ts = train_test_split(df, test_size=0.2, random_state=102)

    data_tr, data_cal = train_test_split(data_train_cal, test_size=0.3, random_state=102)

    #y_train=data_tr["output"]
    #data_tr.drop(["output"],axis=1,inplace=True)
    

    return data_tr, data_cal, data_ts