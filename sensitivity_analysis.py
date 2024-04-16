import pandas as pd
import numpy as np

from settings import tpot_settings
from tpot.gp_deap import _wrapped_cross_val_score
from functools import partial

from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from tpot.builtins import StackingEstimator
from tsftransformers.tsf_transformers import TSF_EWMA, TSF_Abs_Energy, TSF_Sum_Values, TSF_Value_Count, \
    TSF_C3, TSF_Variance, TSF_Mean, TSF_Maximum

from sklearn.linear_model import Ridge, ElasticNet, LassoLars
from sklearn.svm import LinearSVR
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor

if __name__ == '__main__':
    dataset = pd.read_csv('data/measures.csv')
    X = dataset.filter(items=tpot_settings.features_filter)
    y = dataset.filter(items=['pm'])

    pipelines = {
        'seed30': make_pipeline(
    TSF_EWMA(span=3000),
    StackingEstimator(estimator=ElasticNet(l1_ratio=0.01, tol=0.0001)),
    TSF_EWMA(span=6000),
    ExtraTreesRegressor(bootstrap=True, max_depth=30, max_features=0.8, min_samples_leaf=9, min_samples_split=7, n_estimators=100)
),
        'seed32': make_pipeline(
    TSF_EWMA(span=3000),
    TSF_EWMA(span=6000),
    HistGradientBoostingRegressor(l2_regularization=0, learning_rate=0.1, loss="least_squares", max_bins=150, max_depth=10, max_iter=100, min_samples_leaf=9, tol=1e-07, validation_fraction=0.5)
    ),
        'seed40': make_pipeline(
    TSF_EWMA(span=3000),
    TSF_Abs_Energy(window_size=1024),
    LassoLars(normalize=False)
    ),
        'seed54': make_pipeline(
    TSF_EWMA(span=3000),
    TSF_Mean(window_size=512),
    TSF_Mean(window_size=512),
    HistGradientBoostingRegressor(l2_regularization=0, learning_rate=0.1, loss="least_squares", max_bins=150, max_depth=10, max_iter=100, min_samples_leaf=11, tol=1e-07, validation_fraction=0.1)
    ),
        'seed62': make_pipeline(
    TSF_Sum_Values(window_size=128),
    TSF_Mean(window_size=1024),
    TSF_Abs_Energy(window_size=512),
    Ridge()
    )
    }

    partial_wrapped_cross_val_score = partial(
                _wrapped_cross_val_score,
                features=X,
                target=y,
                cv=5,
                scoring_function='neg_mean_squared_error',
                sample_weight=None,
                groups=X.profile_id,
                timeout=tpot_settings.timeout_mins * 60,
                use_dask=False
            )

    # sensitivitätsanalyse für seed 32
    sklearn_pipeline = pipelines['seed32']
    params = sklearn_pipeline.get_params()
    filtered_params = {k: v for k, v in params.items() if k.startswith('histgradient')}

    plus = {
            #'histgradientboostingregressor__l2_regularization': 0.1,
            #'histgradientboostingregressor__learning_rate': 0.11,
            #'histgradientboostingregressor__max_bins': 150+15,
            #'histgradientboostingregressor__max_depth': 11,
            #'histgradientboostingregressor__max_iter': 110,
            #'histgradientboostingregressor__max_leaf_nodes': 34,
            #'histgradientboostingregressor__min_samples_leaf': 10,
            #'histgradientboostingregressor__n_iter_no_change': 11,
            #'histgradientboostingregressor__tol': 1e-08,
            #'histgradientboostingregressor__validation_fraction': 0.6
            #'histgradientboostingregressor__scoring': 'neg_mean_absolute_error'
            }

    minus = {
            #'histgradientboostingregressor__l2_regularization': 0.2,
            #'histgradientboostingregressor__learning_rate': 0.09,
            #'histgradientboostingregressor__max_bins': 150-15,
            #'histgradientboostingregressor__max_depth': 9,
            #'histgradientboostingregressor__max_iter': 90,
            #'histgradientboostingregressor__max_leaf_nodes': 28,
            #'histgradientboostingregressor__min_samples_leaf': 8,
            #'histgradientboostingregressor__n_iter_no_change': 9,
            #'histgradientboostingregressor__tol': 1e-06,
            #'histgradientboostingregressor__validation_fraction': 0.4,
            #'histgradientboostingregressor__scoring': 'neg_root_mean_squared_error'
            }

    to_evaluate_plus = {}
    for k, v in plus.items():
        pipe = clone(sklearn_pipeline)
        setp = {k: v}
        pipe.set_params(**setp)
        to_evaluate_plus[k] = pipe

    to_evaluate_minus = {}
    for k, v in minus.items():
        pipe = clone(sklearn_pipeline)
        setp = {k: v}
        pipe.set_params(**setp)
        to_evaluate_minus[k] = pipe


    # evaluation
    print("-----minus HGB")
    for param, pipeline in to_evaluate_minus.items():
        res = partial_wrapped_cross_val_score(sklearn_pipeline=pipeline)
        scores_per_split = res[4]
        std_ = np.std(scores_per_split)
        print(std_)
        print('Result={}'.format(res))

        lines = [param + ': ' + str(pipeline.get_params()[param]),
                 'Result={}'.format(res),
                 'STD={}'.format(std_),
                 '\n']

        content = [line + '\n' for line in lines]

        with open('sensitivity_analysis_HGB_minus.txt', 'a+') as f:
            f.writelines(content)

    print("-----plus HGB")
    for param, pipeline in to_evaluate_plus.items():
        res = partial_wrapped_cross_val_score(sklearn_pipeline=pipeline)
        scores_per_split = res[4]
        std_ = np.std(scores_per_split)
        print(std_)
        print('Result={}'.format(res))

        lines = [param + ': ' + str(pipeline.get_params()[param]),
                 'Result={}'.format(res),
                 'STD={}'.format(std_),
                 '\n']

        content = [line + '\n' for line in lines]

        with open('sensitivity_analysis_HGB_plus.txt', 'a+') as f:
            f.writelines(content)


    # sensitivitätsanalyse für seed 30
    sklearn_pipeline = pipelines['seed30']
    params = sklearn_pipeline.get_params()
    filtered_params_elastic = {k: v for k, v in params.items() if k.startswith('stackingestimator')}
    filtered_params_et = {k: v for k, v in params.items() if k.startswith('extratrees')}

    plus_elastic = {
        #'stackingestimator__estimator__alpha': 0.8,
        #'stackingestimator__estimator__max_iter': 1100,
        #'stackingestimator__estimator__tol': 1e-5,
        #'stackingestimator__estimator__fit_intercept': False,
        #'stackingestimator__estimator__normalize': True,
        #'stackingestimator__estimator__positive': True,
        #'stackingestimator__estimator__precompute': True,
        #'stackingestimator__estimator__selection': 'random',

    }

    minus_elastic = {
        #'stackingestimator__estimator__alpha': 0.9,
        #'stackingestimator__estimator__max_iter': 900,
        #'stackingestimator__estimator__tol': 1e-3,
    }

    plus_et = {
        #'extratreesregressor__max_depth': 33,
        #'extratreesregressor__max_features': 0.9,
        #'extratreesregressor__min_impurity_decrease': 0.2,
        #'extratreesregressor__min_samples_leaf': 10,
        #'extratreesregressor__min_samples_split': 8,
        #'extratreesregressor__min_weight_fraction_leaf': 0.1,
        #'extratreesregressor__n_estimators': 110,
        'extratreesregressor__bootstrap': False

    }

    minus_et = {
        #'extratreesregressor__oob_score': True,
        #'extratreesregressor__max_depth': 27,
        #'extratreesregressor__max_features': 0.7,
        #'extratreesregressor__min_impurity_decrease': 0.2,
        #'extratreesregressor__min_samples_leaf': 8,
        #'extratreesregressor__min_samples_split': 6,
        #'extratreesregressor__min_weight_fraction_leaf': 0.2,
        #'extratreesregressor__n_estimators': 90
    }

    to_evaluate_plus_elastic = {}
    for k, v in plus_elastic.items():
        pipe = clone(sklearn_pipeline)
        setp = {k: v}
        pipe.set_params(**setp)
        to_evaluate_plus_elastic[k] = pipe

    to_evaluate_minus_elastic = {}
    for k, v in minus_elastic.items():
        pipe = clone(sklearn_pipeline)
        setp = {k: v}
        pipe.set_params(**setp)
        to_evaluate_minus_elastic[k] = pipe

    to_evaluate_minus_et = {}
    for k, v in minus_et.items():
        pipe = clone(sklearn_pipeline)
        setp = {k: v}
        pipe.set_params(**setp)
        to_evaluate_minus_et[k] = pipe

    to_evaluate_plus_et = {}
    for k, v in plus_et.items():
        pipe = clone(sklearn_pipeline)
        setp = {k: v}
        pipe.set_params(**setp)
        to_evaluate_plus_et[k] = pipe


    # evaluation
    print("-----plus elastic ")
    for param, pipeline in to_evaluate_plus_elastic.items():
        res = partial_wrapped_cross_val_score(sklearn_pipeline=pipeline)
        scores_per_split = res[4]
        std_ = np.std(scores_per_split)
        print(std_)
        print('Result={}'.format(res))

        lines = [param + ': ' + str(pipeline.get_params()[param]),
                 'Result={}'.format(res),
                 'STD={}'.format(std_),
                 '\n']

        content = [line + '\n' for line in lines]

        with open('sensitivity_analysis_elastic_plus.txt', 'a+') as f:
            f.writelines(content)

    print("-----minus elastic")
    for param, pipeline in to_evaluate_minus_elastic.items():
        res = partial_wrapped_cross_val_score(sklearn_pipeline=pipeline)
        scores_per_split = res[4]
        std_ = np.std(scores_per_split)
        print(std_)
        print('Result={}'.format(res))

        lines = [param + ': ' + str(pipeline.get_params()[param]),
                 'Result={}'.format(res),
                 'STD={}'.format(std_),
                 '\n']

        content = [line + '\n' for line in lines]

        with open('sensitivity_analysis_elastic_minus.txt', 'a+') as f:
            f.writelines(content)


    print("-----minus et")
    for param, pipeline in to_evaluate_minus_et.items():
        res = partial_wrapped_cross_val_score(sklearn_pipeline=pipeline)
        scores_per_split = res[4]
        std_ = np.std(scores_per_split)
        print(std_)
        print('Result={}'.format(res))

        lines = [param + ': ' + str(pipeline.get_params()[param]),
                 'Result={}'.format(res),
                 'STD={}'.format(std_),
                 '\n']

        content = [line + '\n' for line in lines]

        with open('sensitivity_analysis_et_minus.txt', 'a+') as f:
            f.writelines(content)

    print("-----plus et ")
    for param, pipeline in to_evaluate_plus_et.items():
        res = partial_wrapped_cross_val_score(sklearn_pipeline=pipeline)
        scores_per_split = res[4]
        std_ = np.std(scores_per_split)
        print(std_)
        print('Result={}'.format(res))

        lines = [param + ': ' + str(pipeline.get_params()[param]),
                 'Result={}'.format(res),
                 'STD={}'.format(std_),
                 '\n']

        content = [line + '\n' for line in lines]

        with open('sensitivity_analysis_et_plus.txt', 'a+') as f:
            f.writelines(content)











