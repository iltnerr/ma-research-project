import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

from tsfresh.utilities.dataframe_functions import impute

from sklearn.metrics import mean_squared_error
from sklearn.model_selection._split import GroupKFold

from tpot.builtins import StackingEstimator

from tsfresh.feature_extraction.feature_calculators import abs_energy, mean

from sklearn.linear_model import Ridge, ElasticNet, LassoLars
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

def ewm_prepad(X, s):
    ewm_list = []

    for pid, df in X.groupby('profile_id'):
        df = df.drop(columns=['profile_id'])
        padded_df = df.iloc[[0] * s + list(range(len(df))), :]

        ewm = padded_df.ewm(span=s).mean()
        ewm = ewm.iloc[s:, :]
        ewm = ewm.rename(columns={str(col): str(col) + f"_ewm_{s}" for col in ewm.columns})
        ewm_list.append(ewm)

    ewm_df = pd.concat(ewm_list)
    ewm_df = ewm_df.sort_index()

    # pid nicht droppen, um nächstes feature zu berechnen
    #ret = pd.merge(X.drop(columns=['profile_id']), ewm_df, left_index=True, right_index=True, how="left")
    ret = pd.merge(X, ewm_df, left_index=True, right_index=True, how="left")

    return ret

def preproc(X):
    window_size = 512

    result_df_l = []
    for pid, df in X.groupby('profile_id'):
        df = df.drop(columns=['profile_id'])

        # pd.rolling über alle Zeitreihen
        rolling_results = {}
        for column in df:
            data = df[column]
            rolling_results[column] = data.rolling(window_size).apply(mean)

        result_df = pd.DataFrame(rolling_results)
        result_df_l.append(result_df)

    # DF der rolling Ergebnisse
    ex_feat_df = pd.concat(result_df_l)
    ex_feat_df = ex_feat_df.sort_index()
    impute(ex_feat_df)
    ex_feat_df = ex_feat_df.rename(columns={col: col + f"_tsf" for col in ex_feat_df.columns})

    ret = pd.merge(X.drop(columns=['profile_id']), ex_feat_df, left_index=True,
                   right_index=True, how="left")
    return ret


# Datensatz
dataset = pd.read_csv('data/measures.csv')
features = dataset.filter(items=['profile_id', 'u_q', 'i_d', 'ambient', 'coolant', 'i_q', 'motor_speed', 'torque', 'u_d'])
target = dataset.filter(items=['pm'])

cv = GroupKFold(5)
cv_iter = list(cv.split(features, target, groups=features.profile_id))

fit_pipe2 = True

# CV-Scoring
# Pipeline 1
results = {'true': [], 'preds': [], 'scores': []}
split = 1
print("HistGradientBoostingRegressor")
for train, test in cv_iter:
    print(split)
    X_train = features.iloc[train, :]
    y_train = target.iloc[train, :]
    X_test = features.iloc[test, :]
    y_test = target.iloc[test, :]

    estimator = HistGradientBoostingRegressor(l2_regularization=0, learning_rate=0.1, loss="least_squares", max_bins=150, max_depth=10, max_iter=100, min_samples_leaf=9, tol=1e-07, validation_fraction=0.5)

    # train-split
    X_train = ewm_prepad(X_train, 3000)
    X_train = ewm_prepad(X_train, 6000)
    X_train = X_train.drop(columns=['profile_id'])
    estimator.fit(X_train, y_train)

    # test-split
    X_test = ewm_prepad(X_test, 3000)
    X_test = ewm_prepad(X_test, 6000)
    X_test = X_test.drop(columns=['profile_id'])
    y_preds = estimator.predict(X_test)

    # score
    score, _ = mean_squared_error(y_test, y_preds)
    print(score)

    results['scores'].append(score)
    results['true'].append(y_test['pm'].to_numpy())
    results['preds'].append(y_preds)

    split += 1

print('Scores={}'.format(results['scores']))
print("CV-Score = {}".format(np.average(results['scores'])))


# Pipeline 2
if fit_pipe2:
    results2 = {'true': [], 'preds': [], 'scores': []}
    split = 1
    print("ExtraTreesRegressor")
    for train, test in cv_iter:
        print(split)
        X_train = features.iloc[train, :]
        y_train = target.iloc[train, :]
        X_test = features.iloc[test, :]
        y_test = target.iloc[test, :]
        pid_train = X_train['profile_id']
        pid_test = X_test['profile_id']

        estimator = ExtraTreesRegressor(bootstrap=True, max_depth=30, max_features=0.8, min_samples_leaf=9, min_samples_split=7, n_estimators=100)
        stackingestimator = StackingEstimator(estimator=ElasticNet(l1_ratio=0.01, tol=0.0001))

        # train-split
        X_train = ewm_prepad(X_train, 3000)
        X_train = X_train.drop(columns=['profile_id'])
        X_train = stackingestimator.fit_transform(X_train, y_train)
        X_train = pd.DataFrame(X_train)
        X_train['profile_id'] = pid_train.reset_index(drop=True)
        X_train = ewm_prepad(X_train, 3000)
        X_train = X_train.drop(columns=['profile_id'])
        estimator.fit(X_train, y_train)

        # test-split
        X_test = ewm_prepad(X_test, 3000)
        X_test = X_test.drop(columns=['profile_id'])
        X_test = stackingestimator.transform(X_test)
        X_test = pd.DataFrame(X_test)
        X_test['profile_id'] = pid_test.reset_index(drop=True)
        X_test = ewm_prepad(X_test, 3000)
        X_test = X_test.drop(columns=['profile_id'])
        y_preds = estimator.predict(X_test)

        # score
        score, _ = mean_squared_error(y_test, y_preds)
        print(score)

        results2['scores'].append(score)
        results2['true'].append(y_test['pm'].to_numpy())
        results2['preds'].append(y_preds)

        split += 1

    print('Scores={}'.format(results2['scores']))
    print("CV-Score = {}".format(np.average(results2['scores'])))


# PLOTS
preds = results['preds']
true = results['true']
if fit_pipe2:
    preds2 = results2['preds']
    true2 = results2['true']

# HISTGRADIENT
SMALL_SIZE = 10
BIGGER_SIZE = 10
TICK_SIZE = 0.2

plt.figure(figsize=(6.4, 4.8))
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE-2)  # legend fontsize
plt.rcParams['axes.linewidth'] = 0.1

fig, axs = plt.subplots(3, 2, sharex=True)
for id in [0, 1, 2]:
    t = np.arange(len(preds[id])) / 7200
    # preds vs truth
    axs[id, 0].plot(t, true[id], color='blue', linewidth=0.5, label=r"$\vartheta_{pm}$")
    axs[id, 0].plot(t, preds[id], color='green', linewidth=0.5, label=r"$\hat{\vartheta}_{pm}$")
    axs[id, 0].set_ylabel("Temp. in " + u"\u00b0C")
    axs[id, 0].grid(linewidth=0.25)
    axs[id, 0].xaxis.set_tick_params(width=TICK_SIZE)
    axs[id, 0].yaxis.set_tick_params(width=TICK_SIZE)

    props = dict(facecolor='white', boxstyle="square", pad=0.2, alpha=1, lw=0.3)
    axs[id, 0].text(0.4, 0.95, "MSE: " + str(round(results['scores'][id], 2)) + u" (\u00b0C)\u00b2", fontsize=8,
                    verticalalignment='top', transform=axs[id, 0].transAxes, bbox=props)

    # Error
    error = preds[id]-true[id]
    axs[id, 1].plot(t, error, color='red', linewidth=0.5)
    axs[id, 1].set_ylabel("Temp. in " + u"\u00b0C")
    axs[id, 1].grid(linewidth=0.5)
    axs[id, 1].xaxis.set_tick_params(width=TICK_SIZE)
    axs[id, 1].yaxis.set_tick_params(width=TICK_SIZE)
    axs[id, 1].yaxis.set_major_locator(plticker.MultipleLocator(base=10))
    axs[id, 1].xaxis.set_major_locator(plticker.MultipleLocator(base=5))

    # labels
    axs[2, 0].set_xlabel('Zeit in Stunden')
    axs[2, 1].set_xlabel('Zeit in Stunden')
    axs[0, 1].set_title("Prädiktionsfehler")
    axs[0, 0].set_title("Gemessene und geschätzte Temperatur")
    leg = axs[0, 0].legend()
    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.0)

plt.tight_layout()
plt.savefig("plots/" + "preds_vs_truth-HistGradient.pdf")
plt.clf()

# EXTRATREES
if fit_pipe2:
    plt.figure(figsize=(6.4, 4.8))
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE-2)  # legend fontsize
    plt.rcParams['axes.linewidth'] = 0.1

    fig, axs = plt.subplots(3, 2, sharex=True)
    for id in [0, 1, 2]:
        t = np.arange(len(preds2[id])) / 7200
        # preds vs truth
        axs[id, 0].plot(t, true2[id], color='blue', linewidth=0.5, label=r"$\vartheta_{pm}$")
        axs[id, 0].plot(t, preds2[id], color='brown', linewidth=0.5, label=r"$\hat{\vartheta}_{pm}$")
        axs[id, 0].set_ylabel("Temp. in " + u"\u00b0C")
        axs[id, 0].grid(linewidth=0.25)
        axs[id, 0].xaxis.set_tick_params(width=TICK_SIZE)
        axs[id, 0].yaxis.set_tick_params(width=TICK_SIZE)

        props = dict(facecolor='white', boxstyle="square", pad=0.2, alpha=1, lw=0.3)
        axs[id, 0].text(0.4, 0.95, "MSE: " + str(round(results2['scores'][id], 2)) + u" (\u00b0C)\u00b2", fontsize=8,
                        verticalalignment='top', transform=axs[id, 0].transAxes, bbox=props)

        # Error
        axs[id, 1].plot(t, preds2[id]-true2[id], color='red', linewidth=0.5)
        axs[id, 1].set_ylabel("Temp. in " + u"\u00b0C")
        axs[id, 1].grid(linewidth=0.5)
        axs[id, 1].xaxis.set_tick_params(width=TICK_SIZE)
        axs[id, 1].yaxis.set_tick_params(width=TICK_SIZE)
        axs[id, 1].yaxis.set_major_locator(plticker.MultipleLocator(base=10))
        axs[id, 1].xaxis.set_major_locator(plticker.MultipleLocator(base=5))

        # labels
        axs[2, 0].set_xlabel('Zeit in Stunden')
        axs[2, 1].set_xlabel('Zeit in Stunden')
        axs[0, 1].set_title("Prädiktionsfehler")
        axs[0, 0].set_title("Gemessene und geschätzte Temperatur")
        leg = axs[0, 0].legend()
        for legobj in leg.legendHandles:
            legobj.set_linewidth(1.0)

    plt.tight_layout()
    plt.savefig("plots/" + "preds_vs_truth-ExtraTreesRegressor.pdf")
    plt.clf()

# SCATTER
SMALL_SIZE = 26
BIGGER_SIZE = 28

plt.figure(figsize=(15, 15))
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE-3)  # legend fontsize
plt.rcParams['axes.linewidth'] = 1

min_l = []
max_l = []
if fit_pipe2:
    for id in range(0, 5):
        max_l.append(max(preds2[id] - true2[id]))
        max_l.append(max(preds[id] - true[id]))
        min_l.append(min(preds2[id] - true2[id]))
        min_l.append(min(preds[id] - true[id]))
else:
    for id in range(0, 5):
        max_l.append(max(preds[id] - true[id]))
        min_l.append(min(preds[id] - true[id]))

max_val = max(max_l)
min_val = min(min_l)

for id in range(0, 5):
    ax = plt.subplot(5, 1, id + 1)
    plt.grid(True)
    error = preds[id] - true[id]
    plt.scatter(true[id], error, 3, c="green", label="HistGradientBoosting")
    if fit_pipe2:
        error2 = (preds2[id] - true2[id])
        plt.scatter(true2[id], error2, 3, c="brown", label="ExtraTrees")
    plt.ylim(min_val, max_val)
    plt.ylabel(r"Fehler in " + u"\u00b0C")
    hline = plt.hlines(0, min(true[id]), max(true[id]), colors='black', linestyles="dashed", linewidths=2)
    hline.set_zorder(10)
    if id == 4:
        plt.xlabel(r"$\vartheta_{pm}$ in " + u"\u00b0C")
    if id == 0:
        plt.title("Schätzfehler für den Wertebereich der Temperatur")
        leg = plt.legend()
        for handle in leg.legendHandles:
            handle.set_sizes([80.0])
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=15))

plt.tight_layout()
plt.savefig("plots/" + "error_residuals.png")
plt.clf()


