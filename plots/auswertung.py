import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

from settings.operators_config import tpot_config_pc2

db_path = "data/sql/AutoML.db"

def min_avg(seed_id):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute("SELECT generation, cv_score FROM dauersuche WHERE seed = ? and not cv_score in ('Duplicate(CP)', 'Duplicate(AE)')", (seed_id,))
    result = cursor.fetchall()
    connection.close()

    scores_per_gen = {gen: [] for gen, _ in result}
    for gen, score in result:
        scores_per_gen[gen].append(score)

    max_score_per_gen = [max(scores_per_gen[gen]) for gen, _ in scores_per_gen.items()]
    global_minimum_per_gen = []
    for gen, score in enumerate(max_score_per_gen):
        if gen == 0:
            global_minimum_per_gen.append(score)
        else:
            global_minimum = max(max(global_minimum_per_gen), score)
            global_minimum_per_gen.append(global_minimum)

    global_min = np.multiply(np.array(global_minimum_per_gen), -1)
    minima_per_gen = np.multiply(np.array(max_score_per_gen), -1)

    scores_per_gen_array = []
    for gen, scores_list in scores_per_gen.items():
        scores_per_gen_array.append(np.array([scores_list]))

    avg_per_gen_l = []
    filtered_scores = []
    threshold = -np.inf
    for arr in scores_per_gen_array:
        # filter outliers
        arr = arr[arr > threshold]
        filtered_scores.append(np.multiply(arr, -1))
        avg_per_gen_l.append(np.average(arr))

    avg_per_gen = np.multiply(np.array(avg_per_gen_l), -1)
    return minima_per_gen, avg_per_gen, filtered_scores, global_min

def crashes(seed_id):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute("SELECT generation FROM dauersuche WHERE seed = ? and t_eval_mins in ('CRASHED (memory)', 'CRASHED (ncpus)', 'Timeout')", (seed_id,))
    crashed_pipes = cursor.fetchall()
    cursor.execute("SELECT max(generation) FROM dauersuche WHERE seed = ?", (seed_id,))
    last_gen_number = cursor.fetchone()[0]
    cursor.execute("SELECT generation FROM dauersuche WHERE seed = ? and t_eval_mins != '-'", (seed_id,))
    pipe_count = cursor.fetchall()
    connection.close()

    pipes_l = [i[0] for i in pipe_count]
    pipes_per_gen_d = dict((x, pipes_l.count(x)) for x in set(pipes_l))
    pipes_per_gen_l = [pipes_per_gen_d[gen] for gen in sorted(pipes_per_gen_d.keys())]

    crashes_l = [i[0] for i in crashed_pipes]
    crashes_per_gen_d = dict((x, crashes_l.count(x)) for x in set(crashes_l))

    for gen in range(0, last_gen_number + 1):
        if gen not in crashes_per_gen_d.keys():
            crashes_per_gen_d[gen] = 0

    crashes_per_gen_l = [crashes_per_gen_d[gen] for gen in sorted(crashes_per_gen_d.keys())]
    return np.array(crashes_per_gen_l), np.array(pipes_per_gen_l)

def get_counts(seed_id):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.execute(
        "SELECT generation FROM dauersuche WHERE seed = ? "
        "and cv_score in ('Duplicate(CP)', 'Duplicate(AE)')", (seed_id,))
    dupes = cursor.fetchall()
    dupes_count = len(dupes)

    cursor.execute(
        "SELECT generation, id, cv_score FROM dauersuche WHERE seed = ? "
        "and cv_score not in ('Duplicate(CP)', 'Duplicate(AE)')", (seed_id,))
    scores = cursor.fetchall()
    evaluated_count = len([cv for gen, id, cv in scores if cv != -float(np.inf)])

    cursor.execute(
        "SELECT generation, t_eval_mins FROM dauersuche WHERE seed = ? and "
        "t_eval_mins in ('Timeout', 'CRASHED (ncpus)', 'CRASHED (memory)')", (seed_id,))

    crashed = cursor.fetchall()
    timeout_count = len([cause for gen, cause in crashed if cause == 'Timeout'])
    crashed_mem_count = len([cause for gen, cause in crashed if cause == 'CRASHED (memory)'])
    crashed_cpu_count = len([cause for gen, cause in crashed if cause == 'CRASHED (ncpus)'])

    cursor.execute("SELECT generation, id FROM dauersuche where seed = ?", (seed_id,))
    total_count = len(cursor.fetchall())
    connection.close()

    error_count = total_count - (timeout_count + crashed_cpu_count + crashed_mem_count + evaluated_count + dupes_count)

    counts = {'evaluated' : evaluated_count, 'dupes': dupes_count, 'crashed_mem': crashed_mem_count,
              'crashed_ncpus': crashed_cpu_count, 'timeout': timeout_count, 'errors': error_count}

    for k,v in counts.items():
        print(k + ': ' + str(v))

    print('total: ' + str(sum(counts.values())))
    return counts

def pie_not_evaluated(counts):
    fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(aspect="equal"))

    labels = ["Duplicates",
              "PC2 Shutdown",
              "Timeouts"]

    data = [counts['dupes'], counts['crashed_mem'], counts['timeout']]

    def func(pct, allvals):
        absolute = int(pct / 100. * np.sum(allvals))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                      textprops=dict(color="w"))

    plt.setp(autotexts, size=10, weight="bold")

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(labels[i], xy=(x, y), xytext=(1.15 * np.sign(x), 1.1 * y),
                    horizontalalignment=horizontalalignment, **kw)
    # 1.35, 1.4
    ax.set_title("Nicht evaluierte Pipelines")
    plt.savefig("plots/not evaluated - seed " + str(seed) + plt_format)
    plt.clf()

def pie_counts(counts_per_seed):
    evaluated = 0
    dupes = 0
    crashed_mem = 0
    crashed_ncpus = 0
    timeout = 0
    errors = 0

    for _, counts in counts_per_seed.items():
        evaluated += counts['evaluated']
        dupes += counts['dupes']
        crashed_mem += counts['crashed_mem']
        crashed_ncpus += counts['crashed_ncpus']
        timeout += counts['timeout']
        errors += counts['errors']

    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw=dict(aspect="equal"))

    labels = ["Duplikate",
              "RAM",
              "Timeouts",
              "Error",
              "Ausgewertet"]

    data = [dupes, crashed_mem, timeout, errors, evaluated]

    def func(pct, allvals):
        absolute = int(pct / 100. * np.sum(allvals))
        return "{:.1f}%".format(pct)

    wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                      textprops=dict(color="w"))

    SMALL_SIZE = 4
    MEDIUM_SIZE = 10
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE+2)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize

    plt.setp(autotexts, size=10, weight="bold")

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(labels[i], xy=(x, y), xytext=(1.2 * np.sign(x), 1.4 * y),
                    horizontalalignment=horizontalalignment, **kw)
    # 1.35, 1.4
    ax.set_title("Anteile der abgebrochenen Auswertungen")
    plt.tight_layout()
    plt.savefig("plots/pie.pdf")
    plt.clf()

def feature_usage(config):
    counts = {}
    # init dict
    for feature_string in config.keys():
        if "TSF_" in feature_string:
            feature_string = feature_string[33:]
            counts[feature_string] = 0

    # get counts per seed
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    for feature in counts.keys():
        cursor.execute(
            "SELECT pipeline FROM dauersuche WHERE seed in (30, 32, 40, 54, 60, 62) "
            "and pipeline like ?"
            "and cv_score not in ('Duplicate(CP)', 'Duplicate(AE)')", ("%{}%".format(feature),))
        count_per_feature = len(cursor.fetchall())
        counts[feature] += count_per_feature
    connection.close()
    return counts

def bar_feature_counts(counts):
    sorted_counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1])}

    to_del = []
    i = 0
    acc = 0
    for k, v in sorted_counts.items():
        i += 1
        acc += v
        to_del.append(k)
        if i == 20:
            break

    for key in to_del:
        del sorted_counts[key]

    sorted_counts['TSF_Diverse (20)'] = acc
    sorted_counts = {k: v for k, v in sorted(sorted_counts.items(), key=lambda item: item[1])}


    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels

    fig, ax = plt.subplots()
    feature_labels = list(reversed(list(sorted_counts.keys())))

    shortened_feats = []
    for feat in feature_labels:
        feat = feat[4:]
        if len(feat) > 9 and feat != "Diverse (20)":
            feat = feat[:9] + "."
        shortened_feats.append(feat)

    y_pos = np.arange(len(shortened_feats))
    count = list(reversed(list(sorted_counts.values())))
    performance = np.array(count)

    barlist = ax.barh(y_pos, performance, align='center', height=0.6)
    barlist[1].set_color('g')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(shortened_feats)
    ax.invert_yaxis()
    ax.set_xlabel('Anzahl')
    ax.set_title('Verwendung der TSF_Transformer')
    plt.savefig("plots/feature_usage" + plt_format)
    plt.clf()


seeds = [30, 32, 40, 54, 60, 62]
plt_format = ".pdf"

for seed in seeds:
    # auswertung
    minima, avg_mse, filtered_scores, global_minimum = min_avg(seed)
    crashes_per_gen, pipes_per_gen = crashes(seed)
    counts = get_counts(seed)

    # plots
    BIGGER_SIZE = 10
    SMALL_SIZE = 8

    # Bar: Used Features
    feature_count = feature_usage(tpot_config_pc2)
    bar_feature_counts(feature_count)

    # Pie: Eval/Crash Counts
    counts_all_seeds = {}
    for idx in seeds:
        counts_all_seeds[idx] = get_counts(idx)
    pie_counts(counts_all_seeds)

    # crashes per gen
    fig, ax = plt.subplots()
    plt.title("Verhältnis der abgebrochenen Auswertungen zur Gesamtpopulation", fontsize=13)
    ax.plot(pipes_per_gen, label="Populationsgröße\n(Duplikate ausgenommen)", marker='o')
    ax.plot(crashes_per_gen, label="Ressourcenüberschreitungen", marker='D')
    plt.hlines(5, 0, (len(crashes_per_gen)-1), colors='red', label="10%-Grenzwert", linestyles="dashed")
    plt.hlines(50, 0, (len(crashes_per_gen)-1), colors='black', linestyles="dashed")
    ax.set_ylabel("Anzahl der Individuen", fontsize=SMALL_SIZE+3)
    ax.set_xlabel("Generation", fontsize=SMALL_SIZE+3)
    ax.tick_params(axis='both', labelsize=SMALL_SIZE+3)

    ax.legend()
    plt.savefig("plots/crashes_per_gen-seed-" + str(seed) + plt_format)
    plt.clf()


    # min_avg cv score
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(global_minimum)+1), global_minimum, label="Globales Minimum")
    lw = 0.5
    boxprops = dict(linewidth=lw)
    flierprops = dict(marker='o', markersize=3, linestyle='none', linewidth=lw/2)
    medianprops = dict(linewidth=lw, color='firebrick')
    meanlineprops = dict(linewidth=lw, color='purple')
    whiskerprops = dict(linewidth=lw)
    capprops = dict(linewidth=lw)

    ax.boxplot(filtered_scores, showfliers=True, widths=0.5, boxprops=boxprops, medianprops=medianprops,
               meanprops=meanlineprops, flierprops=flierprops, whiskerprops=whiskerprops, capprops=capprops)
    plt.ylim(0, 300)
    plt.title("Entwicklung des CV Score")
    ax.set_ylabel("MSE in K\u00b2")
    ax.set_xlabel("Generation")
    ax.tick_params(axis='both', labelsize=SMALL_SIZE)
    every_nth = 2
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)

    ax.legend()
    plt.savefig("plots/min_avg_cvscore-seed-" + str(seed) + plt_format)
    plt.clf()
