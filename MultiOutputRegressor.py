import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from target_scaling import check_scalers

class MultiOutputRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
    """Wraps Single-Output Scikit Models and makes them predict on
    n targets by training n models.

    """
    def __init__(self, model, target_cols):
        self.model = model
        self.target_cols = target_cols
        self.multi_model = {t: clone(self.model) for t in target_cols}
        self.multi_scalers = {t: check_scalers(model) for t in target_cols}
        self.input_is_df = False

    def fit(self, X, y, **kwargs):
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        if isinstance(y, pd.DataFrame):
            assert set(self.target_cols) == set(list(y.columns))
            assert all(a==b for a, b in zip(self.target_cols, list(y.columns)))
            self.input_is_df = True
        else:
            assert y.shape[1] == len(self.target_cols), \
                'For MultiOutputRegressor, y.shape must match given target_cols!'
            self.input_is_df = False

        train_kwargs = kwargs.copy()
        for target, model in self.multi_model.items():
            if 'eval_set' in kwargs:
                train_kwargs['eval_set'] = (kwargs['eval_set'][0],
                                            kwargs['eval_set'][1][target])
            if self.input_is_df:
                if self.multi_scalers[target] is not None:
                    # sequentially rescale y with all scalers in the pipeline
                    target_df = pd.DataFrame(y[target])
                    for scaler in self.multi_scalers[target]:
                        target_df = scaler.fit_transform(target_df)
                    print(f'\t\t\tFit on {y[target].name}..')
                    model.fit(X, target_df, **train_kwargs)
                else:
                    print(f'\t\t\tFit on {y[target].name}..')
                    model.fit(X, y[target], **train_kwargs)
            else:
                model.fit(X, y)

    def predict(self, X):
        preds = {}
        for target, model in self.multi_model.items():
            print(f'\t\t\tPredicting {target}..')
            preds[target] = model.predict(X)

            # scale back predictions if necessary
            if self.multi_scalers[target] is not None:
                if len(preds[target].shape) == 1:
                    preds[target] = preds[target].reshape((-1, 1))

                for scaler in reversed(self.multi_scalers[target]):
                    preds[target] = scaler.inverse_transform(preds[target])
                if len(preds[target].shape) == 2:
                    preds[target] = preds[target].reshape(-1)

        ret = pd.DataFrame(preds, index=self.target_cols.index)
        return ret if self.input_is_df else ret.values

    def get_params(self, deep=True):
        if not deep:
            return super().get_params(deep=False)
        else:
            out = {type(model).__name__+'_'+target: model for target, model in self.multi_model.items()}

            for m_name, m in out.copy().items():
                for key, value in m.get_params(deep=True).items():
                    out['{}__{}'.format(m_name, key)] = value

            for key, value in super().get_params(deep=False).items():
                out['{}'.format(key)] = value

            return out

    def set_target_cols(self, target_cols):
        """self.target_cols need to be adjusted to y_test after fitting on y_train"""
        self.target_cols = target_cols
