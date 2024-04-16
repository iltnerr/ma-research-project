import pandas as pd
import custom_functions as cf
import sys

from settings import tpot_settings
from sql_utilities import sql_get_pipecode, sql_set_evaluating, sql_logresult
from tpot.gp_deap import _wrapped_cross_val_score
from functools import partial


def evaluate_pipeline():
    # Dataset
    dataset = pd.read_csv('data/measures.csv')
    X = dataset.filter(items=tpot_settings.features_filter)
    y = dataset.filter(items=tpot_settings.targets_filter)

    # Make Pipeline
    operators_context = cf.pickle_load('opcontext.p')
    idx = int(sys.argv[1])
    pipeline_code, pipeline_string = sql_get_pipecode(idx)

    if pipeline_code is None:
        raise ValueError("Failed to read pipeline_code from the DB!")
    sklearn_pipeline = eval(pipeline_code, operators_context)
    sklearn_pipeline.memory = tpot_settings.cache_dir

    # Fix random state when the operator allows
    set_param_recursive(sklearn_pipeline.steps, 'random_state', tpot_settings.random_state)
    # Setting the seed is needed for XGBoost support because XGBoost currently stores
    # both a seed and random_state, and they're not synced correctly.
    # XGBoost will raise an exception if random_state != seed.
    if 'XGB' in pipeline_code:
        set_param_recursive(sklearn_pipeline.steps, 'seed', tpot_settings.random_state)

    # Evaluate Pipeline
    partial_wrapped_cross_val_score = partial(
                _wrapped_cross_val_score,
                features=X,
                target=y,
                cv=5,
                scoring_function='neg_mean_squared_error',
                sample_weight=None,
                groups=X.profile_id,
                timeout=tpot_settings.timeout_mins * 60,
                use_dask=False,
                pipeline_string=pipeline_string
            )

    sql_set_evaluating(pipeline_string)
    print(f"Evaluating Pipeline: {pipeline_string}")
    print("Pipeline_code:\n")
    print(pipeline_code)
    result = partial_wrapped_cross_val_score(sklearn_pipeline=sklearn_pipeline)

    # Save Results
    if result is not None:
        # result will be None if the evaluation exceeds the timeout limit.
        # In this case the timeout will be logged internally.
        sql_logresult(pipeline_string, result)
        print('Result:\n')
        print(result)
    print('Finished pipeline evaluation.')


def set_param_recursive(pipeline_steps, parameter, value):
    """ Copy from tpot.base.py

    Recursively iterate through all objects in the pipeline and set a given parameter.

    Parameters
    ----------
    pipeline_steps: array-like
        List of (str, obj) tuples from a scikit-learn pipeline or related object
    parameter: str
        The parameter to assign a value for in each pipeline object
    value: any
        The value to assign the parameter to in each pipeline object
    Returns
    -------
    None

    """
    for (_, obj) in pipeline_steps:
        recursive_attrs = ['steps', 'transformer_list', 'estimators']
        for attr in recursive_attrs:
            if hasattr(obj, attr):
                set_param_recursive(getattr(obj, attr), parameter, value)
        if hasattr(obj, 'estimator'):  # nested estimator
            est = getattr(obj, 'estimator')
            if hasattr(est, parameter):
                setattr(est, parameter, value)
        if hasattr(obj, parameter):
            setattr(obj, parameter, value)

if __name__ == '__main__':
    evaluate_pipeline()