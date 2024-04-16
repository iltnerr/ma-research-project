import numpy as np


tpot_config = {
    'sklearn.linear_model.ElasticNet': {
        'l1_ratio': np.arange(0.01, 1.01, 0.05),
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    },

    'sklearn.linear_model.LassoLars': {
        'normalize': [True, False]
    },

    'sklearn.linear_model.Ridge': {
    },

    'sklearn.svm.LinearSVR': {
        'loss': ["epsilon_insensitive", "squared_epsilon_insensitive"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1.]
    },

    'sklearn.ensemble.HistGradientBoostingRegressor': {
        'l2_regularization': [0],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'loss': ['least_squares'],
        'max_bins': [50, 150, 255],
        'max_depth': range(2, 11),
        'max_iter': [100],
        'min_samples_leaf': range(2, 20),
        'tol': [1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04],
        'validation_fraction': [0.01, 0.1, 0.3, 0.5]
    },

    'xgboost.XGBRegressor': {
        'n_estimators': [10, 50, 100, 300],
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'objective': ['reg:squarederror']
    },

    'sklearn.ensemble.AdaBoostRegressor': {
        'n_estimators': [10, 50, 100, 300, 500],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'loss': ["linear", "square", "exponential"]
    },

    'sklearn.tree.DecisionTreeRegressor': {
        'max_depth': range(1, 61),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 11)
    },

    'sklearn.ensemble.ExtraTreesRegressor': {
        'n_estimators': [10, 100, 300, 600],
        'max_depth': [10, 30, 50, 60],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 11),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.RandomForestRegressor': {
        'n_estimators': [10, 50, 100],
        'max_depth': [10, 30, 50, 60],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 11),
        'bootstrap': [True, False]
    },

    'sklearn.neighbors.KNeighborsRegressor': {
        'n_neighbors': range(1, 501),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },

    # Preprocesssors

    'tsftransformers.tsf_transformers.TSF_Variance_Larger_Than_Standard_Deviation': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Has_Duplicate_Max': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Has_Duplicate_Min': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Has_Duplicate': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Sum_Values': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Abs_Energy': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Mean_Abs_Change': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Mean_Change': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Mean_Second_Derivative_Central': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Median': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Mean': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Standard_Deviation': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Variance': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Skewness': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Kurtosis': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Absolute_Sum_Of_Changes': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Longest_Strike_Below_Mean': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Longest_Strike_Above_Mean': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Count_Above_Mean': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Count_Below_Mean': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Last_Location_Of_Maximum': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_First_Location_Of_Maximum': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Last_Location_Of_Minimum': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_First_Location_Of_Minimum': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Percentage_Of_Reoccurring_Datapoints_To_All_Datapoints': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Percentage_Of_Reoccurring_Values_To_All_Values': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Sum_Of_Reoccurring_Values': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Sum_Of_Reoccurring_Data_Points': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Ratio_Value_Number_To_Time_Series_Length': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Maximum': {
        'window_size': [32, 64, 128, 256, 512, 1024]},

    'tsftransformers.tsf_transformers.TSF_Minimum': {
        'window_size': [32, 64, 128, 256, 512, 1024]
    },

    'tsftransformers.tsf_transformers.TSF_EWMA': {
        'span': [3000, 6000, 9000, 12000]
    },

    'tsftransformers.tsf_transformers.TSF_EWMS': {
        'span': [3000, 6000, 9000, 12000]
    },

    'tsftransformers.tsf_transformers.TSF_Time_Reversal_Asymmetry_Statistic': {
        'lag': [1, 2, 3],
        'window_size': [32, 64, 128, 256, 512, 1024]
    },

    'tsftransformers.tsf_transformers.TSF_C3': {
        'lag': [1, 2, 3],
        'window_size': [32, 64, 128, 256, 512, 1024]
    },

    'tsftransformers.tsf_transformers.TSF_Cid_Ce': {
        'normalize': [True, False],
        'window_size': [32, 64, 128, 256, 512, 1024]
    },

    'tsftransformers.tsf_transformers.TSF_Large_Standard_Deviation': {
        'r': np.arange(0.01, 1.01, 0.05),
        'window_size': [32, 64, 128, 256, 512, 1024]
    },

    'tsftransformers.tsf_transformers.TSF_Quantile': {
        'q': [0.1, 0.3, 0.5, 0.7, 0.9],
        'window_size': [32, 64, 128, 256, 512, 1024]
    },

    'tsftransformers.tsf_transformers.TSF_Autocorrelation': {
        'lag': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'window_size': [32, 64, 128, 256, 512, 1024]
    },

    'tsftransformers.tsf_transformers.TSF_Number_CWT_Peaks': {
        'n': [1, 5],
        'window_size': [32, 64, 128, 256, 512, 1024]
    },

    'tsftransformers.tsf_transformers.TSF_Number_Peaks': {
        'n': [1, 3, 5, 10, 50],
        'window_size': [32, 64, 128, 256, 512, 1024]
    },

    'tsftransformers.tsf_transformers.TSF_Binned_Entropy': {
        'max_bins': [3, 5, 10],
        'window_size': [32, 64, 128, 256, 512, 1024]
    },

    'tsftransformers.tsf_transformers.TSF_Change_Quantiles': {
        'ql': [0.0, 0.2, 0.4, 0.6, 0.8],
        'qh': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'isabs': [True, False],
        'f_agg': ['mean', 'var'],
        'window_size': [32, 64, 128, 256, 512, 1024]
    },

    'tsftransformers.tsf_transformers.TSF_Range_Count': {
        'intervall': [(-1, 1), (1000000000000.0, 0), (0, 1000000000000.0)],
        'window_size': [32, 64, 128, 256, 512, 1024]
    },

    'tsftransformers.tsf_transformers.TSF_Number_Crossing_M': {
        'm': [0],
        'window_size': [32, 64, 128, 256, 512, 1024]
    },

    'tsftransformers.tsf_transformers.TSF_Ratio_Beyond_R_Sigma': {
        'r': [0.5, 1.5, 5, 10],
        'window_size': [32, 64, 128, 256, 512, 1024]
    },

    'tsftransformers.tsf_transformers.TSF_Value_Count': {
        'values': [[-1, 0, 1]],
        'window_size': [32, 64, 128, 256, 512, 1024]
    },

    # combiners
    'tsftransformers.tsf_transformers.TSF_Symmetry_Looking': {
        'window_size': [32, 64, 128, 256, 512, 1024]
    },

    'tsftransformers.tsf_transformers.TSF_Index_Mass_Quantile': {
        'q': [[0.2, 0.4, 0.6, 0.8]],
        'window_size': [32, 64, 128, 256, 512, 1024]
     },

    'tsftransformers.tsf_transformers.TSF_Linear_Trend': {
        'attr': [['pvalue', 'rvalue', 'intercept', 'slope', 'stderr']],
        'window_size': [32, 64, 128, 256, 512, 1024]
    },

    'tsftransformers.tsf_transformers.TSF_SPKT_Welch_Density': {
        'coeff': [[2, 5, 8]],
        'window_size': [32, 64, 128, 256, 512, 1024]
    },

    #'tsftransformers.tsf_transformers.TSF_AR_Coefficient': {
    #    'coeff': [[0, 1, 2, 3, 4]],
    #    'window_size': [32, 64, 128, 256, 512, 1024]
    #},

    #'tsftransformers.tsf_transformers.TSF_FFT_Coefficient': {
    #    'window_size': [32, 64, 128, 256, 512, 1024]
    #},

    #'tsftransformers.tsf_transformers.TSF_Augmented_Dickey_Fuller': {
    #    'attr': ['teststat', 'pvalue', 'usedlag'],
    #    'window_size': [32, 64, 128, 256, 512, 1024]
    #},

    'sklearn.preprocessing.Binarizer': {
        'threshold': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.decomposition.FastICA': {
        'tol': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.cluster.FeatureAgglomeration': {
        'linkage': ['ward', 'complete', 'average'],
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    },

    'sklearn.preprocessing.MaxAbsScaler': {
    },

    'sklearn.preprocessing.MinMaxScaler': {
    },

    'sklearn.preprocessing.Normalizer': {
        'norm': ['l1', 'l2', 'max']
    },

    'sklearn.kernel_approximation.Nystroem': {
        'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
        'gamma': np.arange(0.0, 1.01, 0.05),
        'n_components': range(1, 11)
    },

    'sklearn.decomposition.PCA': {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },

    'sklearn.preprocessing.PolynomialFeatures': {
        'degree': [2],
        'include_bias': [False],
        'interaction_only': [False]
    },

    'sklearn.kernel_approximation.RBFSampler': {
        'gamma': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.preprocessing.RobustScaler': {
    },

    'sklearn.preprocessing.StandardScaler': {
    },

    'tpot.builtins.ZeroCount': {
    },

    'tpot.builtins.OneHotEncoder': {
        'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
        'sparse': [False],
        'threshold': [10]
    },

    # Selectors
    'sklearn.feature_selection.SelectFwe': {
        'alpha': np.arange(0, 0.05, 0.001),
        'score_func': {
            'sklearn.feature_selection.f_regression': None
        }
    },

    'sklearn.feature_selection.SelectPercentile': {
        'percentile': range(1, 100),
        'score_func': {
            'sklearn.feature_selection.f_regression': None
        }
    },

    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },

    'sklearn.feature_selection.SelectFromModel': {
        'threshold': np.arange(0, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesRegressor': {
                'n_estimators': [100],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    }

}