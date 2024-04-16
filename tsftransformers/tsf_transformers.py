import pandas as pd
import numpy as np
import tsfresh.defaults

from sklearn.base import BaseEstimator, TransformerMixin
from tsftransformers.tsf_utilities import set_feature_param
from tsfresh.feature_extraction.extraction import extract_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.utils import safe_indexing

from collections import defaultdict


class TSFBase():
    """
    Base class for TSF_Transformers which implements fit() and transform() needed to be compatible with sklearn.
    Every TSF_Transformer needs to preprocess the input DF by adding the 'profile_id' column. This is done with
    the add_p_id() function, which is called during every transform() step.
    """
    def __init__(self, feature, feature_params=None,
                 chunksize=tsfresh.defaults.CHUNKSIZE,
                 n_jobs=tsfresh.defaults.N_PROCESSES, show_warnings=tsfresh.defaults.SHOW_WARNINGS,
                 disable_progressbar=tsfresh.defaults.DISABLE_PROGRESSBAR,
                 impute_function=tsfresh.defaults.IMPUTE_FUNCTION,
                 profile=tsfresh.defaults.PROFILING,
                 profiling_filename=tsfresh.defaults.PROFILING_FILENAME,
                 profiling_sorting=tsfresh.defaults.PROFILING_SORTING):

        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.show_warnings = show_warnings
        self.disable_progressbar = disable_progressbar
        self.impute_function = impute_function
        self.profile = profile
        self.profiling_filename = profiling_filename
        self.profiling_sorting = profiling_sorting

        self.feature_setting = set_feature_param(feature, feature_params)

    def add_p_id(self, X):
        """ Prepare input DF for TSF_Transformer by adding profile_id based on train test split.
        Returns pandas.DataFrame including the 'profile_id' column.
        """
        if self.train_indices is not None and self.test_indices is not None:
            # get subset of profile_id depending on train or test split
            if X.shape[0] == self.train_indices.shape[0]:
                self.is_testsplit = False
                self.profile_id = safe_indexing(self.timeseries_container, self.train_indices)
            elif X.shape[0] == self.test_indices.shape[0]:
                self.is_testsplit = True
                self.profile_id = safe_indexing(self.timeseries_container, self.test_indices)
            else:
                raise ValueError("TSFBase.add_p_id(): Shape of X doesn't fit to train_indices nor test_indices.")

        else:
            # there is no train-test-split
            self.is_testsplit = False
            self.profile_id = self.timeseries_container

        if isinstance(X, np.ndarray):
            X_transformed = pd.DataFrame(data=X, index=self.profile_id.index)
            X_transformed.insert(0, 'profile_id', self.profile_id)
        else:
            assert isinstance(X, pd.DataFrame), "TSFBase: expected pandas.DataFrame, got {}".format(type(X))
            X_transformed = X.copy()
            X_transformed.insert(0, 'profile_id', self.profile_id)

        return X_transformed

    def make_df(self, extracted_features, feature_name):
        """
        Returns pandas DataFrame with extracted features, which will be merged with the input DF.
        There are two approaches for creating the DF:
            a) extracted_features contains single timeseries data per kind and id.
            b) extracted_features contains lists of timeseries data per kind and id, gouped by parameters. This happens
            only for 'combiner' functions. (see tsfresh.feature_extraction.feature_calculators)
        """

        def find_highest_coeff(self, dictofcoeffs):
            """
            For each feature DF in listofcoeffs, find the coefficient for which its abs is max. Then create a DF
            with two columns, where the first column is the abs value and the second column the associated frequency
            of the coefficient.
            """

            def calc_frequency(coeff):
                """
                Calculate frequency from given coefficient.
                Frequency = (1 / (window_size / 2)) * coeff * fs/2, where fs=2Hz
                Returns pd.DataFrame
                """
                if isinstance(coeff, str):
                    coeff = int(coeff)
                return (1 / (self.window_size / 2)) * coeff if not np.isnan(coeff) else np.NaN

            listofdfs = []
            for feature_name, feature_frame in dictofcoeffs.items():
                abs_values = feature_frame.max(axis=1)

                # derive frequency from coefficients with highest abs values
                coefficients = feature_frame.idxmax(axis=1).str[-10:].str.extract('(\d+)')
                frequency = coefficients.applymap(lambda x: calc_frequency(x))

                # combine abs_values and frequency into df
                df = pd.DataFrame()
                df["{}_fft_coefficient_abs".format(feature_name)] = abs_values
                listofdfs.append(df)
                frequency.rename(columns={frequency.columns[0]: "{}_fft_coefficient_freq".format(feature_name)}, inplace=True)
                listofdfs.append(frequency)

            return listofdfs

        # turn extracted_features into list of tuples where one tuple represents a chunk
        # one chunk contains profile_id, kind, and the extracted pd.Series
        it = iter(extracted_features)
        extracted_features = list(zip(it, it, it))

        groupbykind = defaultdict(list)
        groupbykindparam = defaultdict(list)
        # check first entry of extracted_features
        if isinstance(extracted_features[0][2], list):
            # extracted_features is of type b)
            for id, kind, data in extracted_features:
                for entry in data:
                    frame = {'{}_{}_{}'.format(kind, feature_name, entry[0]): entry[1]}
                    df = pd.DataFrame(frame)
                    groupbykindparam[kind].append(df)
                df_kind_param = pd.concat(groupbykindparam[kind], axis=1)
                groupbykind[kind].append((df_kind_param))
                groupbykindparam.clear()

            listofcols = []
            dictofcols = {}
            for key in groupbykind.keys():
                col = pd.concat(groupbykind[key])
                col.sort_index(inplace=True)
                if feature_name == 'fft_coefficient':
                    dictofcols[key] = col
                else:
                    listofcols.append(col)

        else:
            # extracted_features is of type a)
            for id, kind, data in extracted_features:
                frame = {'{}_{}'.format(kind, feature_name): data}
                df = pd.DataFrame(frame)
                groupbykind[kind].append(df)

            listofcols = []
            for key in groupbykind.keys():
                col = pd.concat(groupbykind[key])
                col.sort_index(inplace=True)
                listofcols.append(col)

        if feature_name == 'fft_coefficient':
            # find coefficients for which abs is max
            listofcols = find_highest_coeff(self, dictofcols)

        df = pd.concat(listofcols, axis=1)
        return df

    def drop_columns(self, df):
        """Drop columns which contain a high relative frequency of the same value."""
        p = 0.8 # max allowed relative frequency
        self.to_drop = []
        for col in df.columns:
            highest_frequency = df[col].value_counts(normalize=True, dropna=False).iloc[0]
            if highest_frequency > p:
                self.to_drop.append(col)
        df = df.drop(columns=self.to_drop)
        return df

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = self.add_p_id(X)
        assert isinstance(X, pd.DataFrame), "TSFBase: expected pandas.DataFrame, got {}".format(type(X))
        assert 'profile_id' in X.columns, "TSFBase: DF X has no column named 'profile_id'"

        X_transformed = X.copy()
        extracted_features = extract_features(X_transformed, column_id="profile_id",
                                              disable_progressbar=True,
                                              #n_jobs=0,
                                              default_fc_parameters=self.feature_setting,
                                              window_size=self.window_size)

        # turn extracted features into pd.DataFrame
        df_ext_feat = self.make_df(extracted_features, self.feature)

        # drop unnecessary columns depending on train or test split.
        if self.is_testsplit:
            df_ext_feat = df_ext_feat.drop(columns=self.to_drop)
        else:
            df_ext_feat = self.drop_columns(df_ext_feat)

        if not df_ext_feat.empty:
            assert len(df_ext_feat) == len(X_transformed), \
                "TSFBase: length of df_ext_feat and X_transformed are not equal. " \
                "len(df_ext_feat)={}, len(X_transformed)={}".format(len(df_ext_feat), len(X_transformed))

            # imputation strategy: NaN -> median, -inf -> min, +inf ->max
            impute(df_ext_feat)
            X_transformed = pd.merge(X_transformed.drop(columns=['profile_id']), df_ext_feat, left_index=True, right_index=True, how="left")

        # Make sure that columns have the same order in all splits
        if self.is_testsplit:
            X_transformed = X_transformed[self.column_order]
        else:
            self.column_order = X_transformed.columns.tolist()

        nparray = X_transformed.values
        return nparray


"""
Init Classes for TSF_Transformers. 
A TSF_Transformer is supposed to calculate a single feature with respective hyperparameters chosen by TPOT.
Every TSF_Transformer inherits from TSFBase as a sklearn compatible transformer.
For further descriptions of specific features see: https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html

Note: Classes should follow this naming style: TSF_<feature_name>
"""

# 'Simple' features
class TSF_Variance_Larger_Than_Standard_Deviation(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Has_Duplicate_Max(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Has_Duplicate_Min(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Has_Duplicate(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Sum_Values(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Abs_Energy(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Mean_Abs_Change(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Mean_Change(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Mean_Second_Derivative_Central(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Median(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Mean(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Length(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Standard_Deviation(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Variance(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Skewness(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Kurtosis(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Absolute_Sum_Of_Changes(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Longest_Strike_Below_Mean(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Longest_Strike_Above_Mean(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Count_Above_Mean(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Count_Below_Mean(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Last_Location_Of_Maximum(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_First_Location_Of_Maximum(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Last_Location_Of_Minimum(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_First_Location_Of_Minimum(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Percentage_Of_Reoccurring_Datapoints_To_All_Datapoints(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Percentage_Of_Reoccurring_Values_To_All_Values(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Sum_Of_Reoccurring_Values(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Sum_Of_Reoccurring_Data_Points(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Ratio_Value_Number_To_Time_Series_Length(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Maximum(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Minimum(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, feature_params=None, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = feature_params

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

# With params
class TSF_EWMA(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, span, train_indices=None, test_indices=None, timeseries_container=None):

        self.window_size = None
        self.span = span
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = {'span': self.span}

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_EWMS(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, span, train_indices=None, test_indices=None, timeseries_container=None):

        self.window_size = None
        self.span = span
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = {'span': self.span}

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_EWMCORR(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, span, train_indices=None, test_indices=None, timeseries_container=None):

        self.window_size = None
        self.span = span
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = {'span': self.span}

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_EWMCOV(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, span, train_indices=None, test_indices=None, timeseries_container=None):

        self.window_size = None
        self.span = span
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = {'span': self.span}

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Time_Reversal_Asymmetry_Statistic(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, lag, window_size, train_indices=None, test_indices=None, timeseries_container=None):

        self.window_size = window_size
        self.lag = lag
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = {'lag': self.lag}

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_C3(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, lag, window_size, train_indices=None, test_indices=None, timeseries_container=None):

        self.window_size = window_size
        self.lag = lag
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = {'lag': self.lag}

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Cid_Ce(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, normalize, window_size, train_indices=None, test_indices=None, timeseries_container=None):

        self.normalize = normalize
        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = {'normalize': self.normalize}

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Large_Standard_Deviation(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, r, window_size, train_indices=None, test_indices=None, timeseries_container=None):

        self.r = r
        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = {'r': self.r}

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Quantile(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, q, window_size, train_indices=None, test_indices=None, timeseries_container=None):

        self.q = q
        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = {'q': self.q}

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Autocorrelation(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, lag, window_size, train_indices=None, test_indices=None, timeseries_container=None):

        self.lag = lag
        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = {'lag': self.lag}

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Number_CWT_Peaks(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, n, window_size, train_indices=None, test_indices=None, timeseries_container=None):

        self.n = n
        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = {'n': self.n}

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Number_Peaks(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, n, window_size, train_indices=None, test_indices=None, timeseries_container=None):

        self.n = n
        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = {'n': self.n}

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Binned_Entropy(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, max_bins, window_size, train_indices=None, test_indices=None, timeseries_container=None):

        self.max_bins = max_bins
        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = {'max_bins': self.max_bins}

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Change_Quantiles(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, ql, qh, isabs, f_agg, window_size, train_indices=None,
                 test_indices=None, timeseries_container=None):

        self.ql = ql
        self.qh = qh
        self.isabs = isabs
        self.f_agg = f_agg
        self.window_size = window_size
        assert self.ql < self.qh, "TSF_Change_Quantiles: expected ql < qh. Got ql={}, qh={}.".format(ql, qh)
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = {'ql': self.ql, 'qh': self.qh, 'isabs': self.isabs, 'f_agg': self.f_agg}

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Value_Count(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, values, window_size, train_indices=None, test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.values = values

        TSFBase.__init__(self, self.feature, self.values)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Range_Count(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, intervall, window_size, train_indices=None, test_indices=None, timeseries_container=None):

        self.intervall = intervall
        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = {'min': self.intervall[0], 'max': self.intervall[1]}

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Max_Langevin_Fixed_Point(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, m, r, window_size, train_indices=None, test_indices=None, timeseries_container=None):

        self.m = m
        self.r = r
        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = {'m': self.m, 'r': self.r}

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Number_Crossing_M(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, m, window_size, train_indices=None, test_indices=None, timeseries_container=None):

        self.m = m
        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = {'m': self.m}

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Ratio_Beyond_R_Sigma(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, r, window_size, train_indices=None, test_indices=None, timeseries_container=None):

        self.r = r
        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = {'r': self.r}

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices


# 'Combiner' features
class TSF_Symmetry_Looking(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, train_indices=None, test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.r = [0.05, 0.25, 0.45, 0.65, 0.85]

        TSFBase.__init__(self, self.feature, self.r)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Linear_Trend(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, attr, window_size, train_indices=None, test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.attr = attr

        TSFBase.__init__(self, self.feature, self.attr)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Index_Mass_Quantile(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, q, window_size, train_indices=None, test_indices=None, timeseries_container=None):

        self.q = q
        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]

        TSFBase.__init__(self, self.feature, self.q)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_SPKT_Welch_Density(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, coeff, window_size, train_indices=None, test_indices=None, timeseries_container=None):

        self.coeff = coeff
        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]

        TSFBase.__init__(self, self.feature, self.coeff)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_AR_Coefficient(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, coeff, window_size, train_indices=None, test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.coeff = coeff

        TSFBase.__init__(self, self.feature, self.coeff)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_FFT_Coefficient(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, window_size, train_indices=None, test_indices=None, timeseries_container=None):

        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]

        TSFBase.__init__(self, self.feature)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices

class TSF_Augmented_Dickey_Fuller(BaseEstimator, TransformerMixin, TSFBase):
    def __init__(self, attr, window_size, train_indices=None, test_indices=None, timeseries_container=None):

        self.attr = attr
        self.window_size = window_size
        # generate feature string from class name
        feature = str(__class__.__name__).lower()
        self.feature = feature[4:]
        self.feature_params = {'attr': self.attr}

        TSFBase.__init__(self, self.feature, self.feature_params)

        # attributes to be set during pipeline preprocessing
        self.timeseries_container = timeseries_container
        self.train_indices = train_indices
        self.test_indices = test_indices