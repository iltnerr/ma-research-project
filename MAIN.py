from tpot import TPOTRegressor
import pandas as pd
import numpy as np
import custom_functions as cf
from settings import tpot_settings, operators_config


dataset = pd.read_csv("data/measures.csv")
X = dataset.filter(items=tpot_settings.features_filter)
y = dataset.filter(items=tpot_settings.targets_filter)


tpot = TPOTRegressor(
    generations=100,
    population_size=50,
    warm_start=True,
    max_eval_time_mins=tpot_settings.timeout_mins,
    n_jobs=1,                                   # Dont change n_jobs !!
    scoring='neg_mean_squared_error',
    config_dict=operators_config.tpot_config,
    max_time_mins=None,                         # Time limit for TPOT run
    cv=5,
    offspring_size=None,                        # None -> offspring_size = population_size
    mutation_rate=0.9,
    crossover_rate=0.1,
    verbosity=2,
    random_state=tpot_settings.random_state,
    memory=None,                                # Cache dir. This property is set in pipeline_evaluation.py
    use_dask=False,
    periodic_checkpoint_folder=tpot_settings.checkpoints_dir,
    early_stop=5,                            # None or int, number of generations w/o progress before closing the run
    subsample=1,
    template=None
)

tpot.fit(X, y)

print('================================')
print('Dauersuche mittels TPOT beendet.')