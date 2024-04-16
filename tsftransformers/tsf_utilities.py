import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from tsfresh.feature_extraction import EfficientFCParameters


def prepoc_pipeline(sklearn_pipeline, X, train_indices, test_indices):
    """
     Preprocessing of Pipelines in order to use TSF_Transformers which need a 'profile_id' column in the dataset.
     This function assigns the attributes timeseriess_container, train_indices, test_indices
     for every TSF_Transformer within the pipeline.
    """
    paramdict = {}
    for key in sklearn_pipeline.get_params().keys():
        if 'timeseries_container' in key:
            paramdict[key] = X['profile_id']
        if train_indices is not None and 'train_indices' in key:
            paramdict[key] = train_indices
        if test_indices is not None and 'test_indices' in key:
            paramdict[key] = test_indices

    if len(paramdict) != 0:
        sklearn_pipeline.set_params(**paramdict)
    return sklearn_pipeline


def set_feature_param(feature, feature_params=None):
    """Returns TSFresh feature settings Object."""

    def fft_coeff_setting(feature_setting):
        """Only calculate the first 10 fft coefficients. Since the window size is at least 32, the length of the
        window will be long enough to calculate all 10 coefficients. For smaller window sizes the following formula
        can be used to calculate how many coefficients can be calculated by the fft:
        length = (window_size / 2) + 1 if (window_size % 2) == 0 else (window_size + 1) / 2
        """
        settings = [{'coeff': coeff} for coeff in range(0,10)]
        feature_setting['fft_coefficient'] = settings
        return feature_setting


    assert isinstance(feature, str), 'set_feature_param(): feature to be set is no string.'
    feature_setting = EfficientFCParameters()
    feature_setting.clear()
    if feature_params is not None:
        if isinstance(feature_params, list):
            feature_setting[feature] = [{'param': param} for param in feature_params]
        else:
            feature_setting[feature] = feature_params
    else:
        if feature == 'fft_coefficient':
            feature_setting = fft_coeff_setting(feature_setting)
        else:
            feature_setting[feature] = None

    return feature_setting