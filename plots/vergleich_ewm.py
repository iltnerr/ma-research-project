import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tsfresh.utilities.dataframe_functions import impute

from sklearn.metrics import mean_squared_error
from sklearn.model_selection._split import GroupKFold

from sklearn.preprocessing import StandardScaler, PolynomialFeatures, Binarizer
from tpot.builtins import StackingEstimator
from sklearn.feature_selection import VarianceThreshold

from tsfresh.feature_extraction.feature_calculators import abs_energy, mean, c3, sum_values, skewness

from sklearn.linear_model import Ridge, ElasticNet, LassoLars
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

def ewm_prepad(X):
    s = 3000
    ewm_list = []

    for pid, df in X.groupby('profile_id'):
        df = df.drop(columns=['profile_id'])
        padded_df = df.iloc[[0] * s + list(range(len(df))), :]

        ewm = padded_df.ewm(span=s).mean()
        ewm = ewm.iloc[s:, :]
        ewm = ewm.rename(columns={col: col + f"_ewm_{s}" for col in ewm.columns})
        ewm_list.append(ewm)

    ewm_df = pd.concat(ewm_list)
    ewm_df = ewm_df.sort_index()

    # pid nicht droppen, um nächstes feature zu berechnen
    # ret = pd.merge(X.drop(columns=['profile_id']), ewm_df, left_index=True, right_index=True, how="left")
    ret = pd.merge(X, ewm_df, left_index=True, right_index=True, how="left")

    return ret

def preproc(X, func):
    window_size = 1024

    result_df_l = []
    for pid, df in X.groupby('profile_id'):
        df = df.drop(columns=['profile_id'])

        # pd.rolling über alle Zeitreihen
        rolling_results = {}
        for column in df:
            data = df[column]
            rolling_results[column] = data.rolling(window_size).apply(func)

        result_df = pd.DataFrame(rolling_results)
        result_df_l.append(result_df)

    # DF der rolling Ergebnisse
    ex_feat_df = pd.concat(result_df_l)
    ex_feat_df = ex_feat_df.sort_index()
    impute(ex_feat_df)
    ex_feat_df = ex_feat_df.rename(columns={col: col + f"_tsf" for col in ex_feat_df.columns})

    #ret = pd.merge(X.drop(columns=['profile_id']), ex_feat_df, left_index=True,
    #               right_index=True, how="left")

    return ex_feat_df

def preprocc3(X, lag):
    window_size = 1024

    result_df_l = []
    for pid, df in X.groupby('profile_id'):
        df = df.drop(columns=['profile_id'])

        # pd.rolling über alle Zeitreihen
        rolling_results = {}
        for column in df:
            data = df[column]
            rolling_results[column] = data.rolling(window_size).apply(c3, args=(lag,))

        result_df = pd.DataFrame(rolling_results)
        result_df_l.append(result_df)

    # DF der rolling Ergebnisse
    ex_feat_df = pd.concat(result_df_l)
    ex_feat_df = ex_feat_df.sort_index()
    impute(ex_feat_df)
    ex_feat_df = ex_feat_df.rename(columns={col: col + f"_tsf" for col in ex_feat_df.columns})

    #ret = pd.merge(X.drop(columns=['profile_id']), ex_feat_df, left_index=True,
    #               right_index=True, how="left")

    return ex_feat_df


# Datensatz
dataset = pd.read_csv('data/measures.csv')
features = dataset.filter(items=['profile_id', 'u_q', 'i_d', 'ambient', 'coolant', 'i_q', 'motor_speed', 'torque', 'u_d'])
target = dataset.filter(items=['pm'])

cv = GroupKFold(5)
cv_iter = list(cv.split(features, target, groups=features.profile_id))

train, test = cv_iter[0]
X_train = features.iloc[train, :]
y_train = target.iloc[train, :]
X_test = features.iloc[test, :]
y_test = target.iloc[test, :]


X_train = ewm_prepad(X_train.drop(columns=['i_d', 'ambient', 'i_q', 'torque', 'u_d', 'u_q']))
mean_ = preproc(X_train, mean)

"""
X_test_ewm_vgl = ewm_prepad(X_test.drop(columns=['i_d', 'coolant', 'i_q', 'torque', 'u_d', 'u_q']))
abs_ewm_vgl = preproc(X_test_ewm_vgl, abs_energy)
mean_ewm_vgl = preproc(X_test_ewm_vgl, mean)
"""
# scaling
"""
X_test_ewm_vgl = StandardScaler().fit_transform(X_test_ewm_vgl.drop(columns=['profile_id']))
abs_ewm_vgl = StandardScaler().fit_transform(abs_ewm_vgl)
mean_ewm_vgl = StandardScaler().fit_transform(mean_ewm_vgl)
"""

ewma = X_train.drop(columns=['profile_id'])
ewma = StandardScaler().fit_transform(ewma)
mean_ = StandardScaler().fit_transform(mean_)
y_train = StandardScaler().fit_transform(y_train)


# plots
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 15


# DISTPLOT
plt.figure(figsize=(6.4, 6))
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE-4)  # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE-3)  # legend fontsize

kde = True
plt.subplot(2, 1, 1)
plt.title("Motordrehzahl")
sns.distplot(y_train[:, 0], color='black', label='pm', kde=kde)
sns.distplot(ewma[:, 1], color='purple', label='motor_speed', kde=kde)
sns.distplot(ewma[:, 3], color='green', label='motor_speed_EWMA', kde=kde)
sns.distplot(mean_[:, 1], color='orange', label='motor_speed_Mean', kde=kde)
plt.ylabel("Wahrscheinlichkeitsdichte")
plt.xlabel("Drehzahl (standardisiert)")
plt.legend()

plt.subplot(2, 1, 2)
plt.title("Kühlmitteltemperatur")
sns.distplot(y_train[:, 0], color='black', label='pm', kde=kde)
sns.distplot(ewma[:, 0], color='purple', label='coolant', kde=kde)
sns.distplot(ewma[:, 2], color='green', label='coolant_EWMA', kde=kde)
sns.distplot(mean_[:, 0], color='orange', label='coolant_Mean', kde=kde)
plt.ylabel("Wahrscheinlichkeitsdichte")
plt.xlabel("Temperatur (standardisiert)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/distplot.pdf")
plt.clf()

"""
# ewm vergleich
plt.figure(figsize=(6.4, 6))
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE-4)  # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE-2)  # legend fontsize

t = np.arange(len(X_test_ewm_vgl)) / 7200
plt.subplot(2, 1, 1)
plt.title("Umgebungstemperatur")
plt.plot(t, X_test_ewm_vgl[:, 0], color='orange', label='ambient', linewidth=0.5)
plt.plot(t, X_test_ewm_vgl[:, 2], color='green', label='ambient_EWMA')
plt.plot(t, abs_ewm_vgl[:, 0], color='purple', label='ambient_Abs_Energy')
#plt.plot(t, mean_ewm_vgl[:, 0], color='purple', label='ambient_Mean')
plt.ylabel("Temperatur\n(standardisiert)")
plt.xlabel("Zeit in Stunden")
leg = plt.legend()
for legobj in leg.legendHandles:
    legobj.set_linewidth(1.0)

plt.subplot(2, 1, 2)
plt.title("Motordrehzahl")
plt.plot(t, X_test_ewm_vgl[:, 1], color='orange', label='motor_speed', linewidth=0.5)
plt.plot(t, X_test_ewm_vgl[:, 3], color='green', label='motor_speed_EWMA')
plt.plot(t, abs_ewm_vgl[:, 1], color='purple', label='motor_speed_Abs_Energy')
#plt.plot(t, mean_ewm_vgl[:, 1], color='purple', label='motor_speed_Mean')
plt.ylabel("Drehzahl\n(standardisiert)")
plt.xlabel("Zeit in Stunden")
leg = plt.legend()
for legobj in leg.legendHandles:
    legobj.set_linewidth(1.0)

plt.tight_layout()
plt.savefig("plots/vergleich_ewm.pdf")
plt.clf()
"""