import numpy as np
import sqlite3
import os
import time
import logging
from datetime import datetime
from os.path import join
from settings import tpot_settings

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
db_path = os.environ['PC2PFS'] + "/hpc-prf-neunet/riltner/AutoML.db"

# Logging for MAIN.py
logfile = join(ROOT_PATH, 'main') + '.log'
logging.basicConfig(level=logging.INFO, filename=logfile, format="%(asctime)-15s %(levelname)-8s %(message)s")

def error_info(e, func_name, main=False):
    print('\n----------Catched SQL Error in ' + func_name + ': ' + str(e))
    print('Try again in 10 seconds.')

    if main:
        msg = '\n----------Catched SQL Error in ' + func_name + ': ' + str(e)
        logging.info(msg)
        logging.info('Try again in 10 seconds.')

    time.sleep(10)

def find_pending():
    try:
        with sqlite3.connect(db_path, timeout=120.0) as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT status FROM evaluate_generation")
            results = cursor.fetchall()

            pending_ids = []
            for idx, row in enumerate(results):
                if row[0] == 'PENDING':
                    pending_ids.append(idx)

            return pending_ids
    except sqlite3.OperationalError as e:
        error_info(e, 'find_pending')
        return find_pending()
    except Exception as e:
        print(e)

def sql_set_pending(pipe_string):
    """
    Set status of allocated pipeline to 'PENDING' so that only pipelines with this status can be fetched for evaluation.
    """
    try:
        with sqlite3.connect(db_path, timeout=120.0) as connection:
            cursor = connection.cursor()
            sql_command = """
                                UPDATE evaluate_generation
                                SET status = 'PENDING'
                                WHERE status = 'UNREAD' AND pipeline_string = ?
                                """
            data_tuple = (pipe_string, )
            cursor.execute(sql_command, data_tuple)
    except sqlite3.OperationalError as e:
        error_info(e, 'sql_set_pending', main=True)
        sql_set_pending(pipe_string)
    except Exception as e:
        print(e)

def sql_summarize(best_pipeline, score, seed, duration):
    """At the end of the TPOT run, log the best pipeline in a seperate table."""

    def sql_createtable_summary(cursor):
        """ Check whether table exists in db and create it if it does not.
        """

        create_table_command = """
        CREATE TABLE summary (
        pipeline VARCHAR(180),
        cv_score FLOAT,
        seed INT,
        duration_hours FLOAT,
        times TIMESTAMP);"""

        # get the count of tables with the name
        cursor.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='summary' ''')

        # if the count is 1, then table exists
        if cursor.fetchone()[0] == 1:
            # db exists already
            pass
        else:
            cursor.execute(create_table_command)
            print("table 'summary' created.")

    with sqlite3.connect(db_path, timeout=120.0) as connection:
        cursor = connection.cursor()
        sql_createtable_summary(cursor)
        # Log best pipeline
        cursor.execute("""INSERT INTO summary (pipeline, cv_score, seed, duration_hours, times) VALUES (?,?,?,?,?);""",
                       (str(best_pipeline), score, seed, duration, datetime.now()))

def sql_check_for_crash(timeout):
    """If an evaluation takes longer than 'timeout' to respond, something unexpected happened during
     the evaluation, which caused a crash. In this case a result has to be set explicitly with this function.
    """
    def select_evaluating():
        try:
            with sqlite3.connect(db_path, timeout=120.0) as connection:
                cursor = connection.cursor()
                cursor.execute("SELECT pipeline_string, starttime FROM evaluate_generation WHERE status = 'EVALUATING'")
                result = cursor.fetchall()
                return result
        except sqlite3.OperationalError as e:
            error_info(e, 'select_evaluating', main=True)
            return select_evaluating()
        except Exception as e:
            print(e)

    def set_result_crashed(no_response):
        """Set result for crashed pipelines."""
        try:
            with sqlite3.connect(db_path, timeout=120.0) as connection:
                # connection.set_trace_callback(logging.info)
                cursor = connection.cursor()
                sql_command = """INSERT INTO pipeline_results (pipeline, cv_score, pm_cv, stator_yoke_cv, stator_tooth_cv,
                 stator_winding_cv, t_eval_mins, starttime)
                VALUES (?,?,?,?,?,?,?,?);"""
                cursor.executemany(sql_command, no_response)
        except sqlite3.OperationalError as e:
            error_info(e, 'set_result_crashed', main=True)
            set_result_crashed(no_response)
        except Exception as e:
            print(e)

    evaluating_pipelines = select_evaluating()

    # list of crashed pipelines
    no_response = []
    for row in evaluating_pipelines:
        pipeline_string, starttime_string = row
        starttime = datetime.strptime(starttime_string, '%Y-%m-%d %H:%M:%S.%f')
        eval_time = datetime.now() - starttime
        if eval_time.seconds > (timeout * 60):
            no_response.append((pipeline_string, -float('inf'), '-', '-', '-', '-', "CRASHED", starttime))

    # set results for crashed pipelines
    set_result_crashed(no_response)

def sql_clear_tables():
    """Clears tables 'pipeline_results' and 'evaluate_generation' as preparation for the evaluation"""

    def sql_createtable_pipeline_results(cursor):
        """ Check whether table exists in db and create it if it does not.
        """

        create_table_command = """
        CREATE TABLE pipeline_results (
        pipeline VARCHAR(180),
        cv_score FLOAT,
        pm_cv FLOAT, 
        stator_yoke_cv FLOAT, 
        stator_tooth_cv FLOAT,
        stator_winding_cv FLOAT, 
        t_eval_mins FLOAT,
        starttime TIMESTAMP);"""

        # get the count of tables with the name
        cursor.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='pipeline_results' ''')

        # if the count is 1, then table exists
        if cursor.fetchone()[0] == 1:
            # db exists already
            pass
        else:
            cursor.execute(create_table_command)
            print("table 'pipeline_results' created.")

    def sql_createtable_evaluate_generation(cursor):
        """ Check whether table exists in db and create it if it does not.
        """

        create_table_command = """
        CREATE TABLE evaluate_generation (
        pipeline_code VARCHAR(180),
        pipeline_string VARCHAR(180),
        status VARCHAR(180), 
        starttime TIMESTAMP);"""

        # get the count of tables with the name
        cursor.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='evaluate_generation' ''')

        # if the count is 1, then table exists
        if cursor.fetchone()[0] == 1:
            # db exists already
            pass
        else:
            cursor.execute(create_table_command)
            print("table 'evaluate_generation' created.")

    with sqlite3.connect(db_path, timeout=120.0) as connection:
        cursor = connection.cursor()
        sql_createtable_pipeline_results(cursor)
        sql_createtable_evaluate_generation(cursor)
        cursor.execute("DELETE FROM pipeline_results")
        cursor.execute("DELETE FROM evaluate_generation")

def sql_evaluating_count():
    """Returns count of pipelines currently being evaluated."""
    try:
        with sqlite3.connect(db_path, timeout=120.0) as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT pipeline_string FROM evaluate_generation WHERE status IN ('EVALUATING', 'PENDING')")
            result = cursor.fetchall()
            return len(result)
    except sqlite3.OperationalError as e:
        error_info(e, 'sql_evaluating_count', main=True)
        return sql_evaluating_count()
    except Exception as e:
        print(e)

def sql_get_results(resultdict):
    """
    Read 'pipeline_results' for results of finished pipelines for further processing during the TPOT run. After reading,
    the entries will be deleted as preperation for the next call to this function.
    Also update 'evaluate_generation' in case 'pipeline_results' contains results.

    Returns a dict containing the string of a pipeline as key and the result as value.
    """
    def fetch_results():
        try:
            with sqlite3.connect(db_path, timeout=120.0) as connection:
                cursor = connection.cursor()
                cursor.execute("SELECT pipeline, cv_score, pm_cv, stator_yoke_cv, stator_tooth_cv, "
                               "stator_winding_cv, t_eval_mins, starttime FROM pipeline_results")
                results = cursor.fetchall()
                return results
        except sqlite3.OperationalError as e:
            error_info(e, 'fetch_results', main=True)
            return fetch_results()
        except Exception as e:
            print(e)

    def clear_pipeline_results_table(results):
        try:
            with sqlite3.connect(db_path, timeout=120.0) as connection:
                connection.set_trace_callback(logging.info)
                cursor = connection.cursor()
                for res in results:
                    cursor.execute("""Delete from pipeline_results where pipeline = ?""", (res[0],))
        except sqlite3.OperationalError as e:
            error_info(e, 'clear_pipeline_results_table', main=True)
            clear_pipeline_results_table(results)
        except Exception as e:
            print(e)

    def update_evaluate_generation(to_update):
        try:
            with sqlite3.connect(db_path, timeout=120.0) as connection:
                # connection.set_trace_callback(logging.info)
                cursor = connection.cursor()
                sql_command = """
                                    UPDATE evaluate_generation
                                    SET status = 'FINISHED'
                                    WHERE pipeline_string = ?
                                    """
                cursor.executemany(sql_command, to_update)
        except sqlite3.OperationalError as e:
            error_info(e, 'update_evaluate_generation', main=True)
            update_evaluate_generation(to_update)
        except Exception as e:
            print(e)

    results = fetch_results()

    # find new results, which were not fetched before
    new_results = []
    for res in results:
        if res[0] not in resultdict.keys():
            new_results.append(res)

    # update dict of results with the new results and prepare list with status to update
    to_update = []
    for res in new_results:
        resultdict[res[0]] = res[1:]
        to_update.append((res[0], ))

    # clear 'pipeline_results' from fetched results
    # clear_pipeline_results_table(results)

    # update 'evaluate_generation' in case new results were fetched from 'pipeline_results'
    update_evaluate_generation(to_update)
    return resultdict

def sql_logresult(pipeline_string, result):
    """Log result of a pipeline evaluation."""

    try:
        with sqlite3.connect(db_path, timeout=120.0) as connection:
            connection.set_trace_callback(print)
            cursor = connection.cursor()
            sql_command = """INSERT INTO pipeline_results (pipeline, cv_score, pm_cv, stator_yoke_cv, stator_tooth_cv, 
            stator_winding_cv, t_eval_mins, starttime)
            VALUES (?,?,?,?,?,?,?,?);"""

            cv_score, t_start, t_eval, ind_scores = result
            data_tuple = (pipeline_string, cv_score, ind_scores[0], ind_scores[1], ind_scores[2], ind_scores[3],
                          t_eval, t_start)
            cursor.execute(sql_command, data_tuple)
            print("Logged result in the DB.")
    except sqlite3.OperationalError as e:
        error_info(e, 'sql_logresult')
        sql_logresult(pipeline_string, result)
    except Exception as e:
        print(e)

def sql_get_pipecode(idx):
    """Get the string and code of a pipeline to evaluate it."""

    try:
        with sqlite3.connect(db_path, timeout=120.0) as connection:
            connection.set_trace_callback(print)
            cursor = connection.cursor()
            cursor.execute("SELECT pipeline_code, pipeline_string FROM evaluate_generation")
            row = cursor.fetchall()
            print("Fetched string and code of pipeline.")
            return row[idx] if row is not None else (None, None)
    except sqlite3.OperationalError as e:
        error_info(e, 'sql_get_pipecode')
        return sql_get_pipecode(idx)
    except Exception as e:
        print(e)

def sql_set_evaluating(string):
    try:
        with sqlite3.connect(db_path, timeout=120.0) as connection:
            connection.set_trace_callback(print)
            cursor = connection.cursor()
            sql_command = """
                            UPDATE evaluate_generation
                            SET status = 'EVALUATING', starttime = ?
                            WHERE pipeline_string = ?
                            """

            cursor.execute(sql_command, (datetime.now(), string))
            print('Status has been set to "EVALUATING".')
    except sqlite3.OperationalError as e:
        error_info(e, 'sql_set_evaluating')
        sql_set_evaluating(string)
    except Exception as e:
        print(e)


def sql_pipecodes_to_db(pipeline_codes, pipeline_strings):
    """Output codes and strings of all pipelines in the generation to db so that other processes can fetch and
    evaluate them pipeline."""

    with sqlite3.connect(db_path, timeout=120.0) as connection:
        cursor = connection.cursor()
        sql_command = """INSERT INTO evaluate_generation (pipeline_code, pipeline_string, status, starttime)
        VALUES (?,?,?,?);"""
        for code, string in zip(pipeline_codes, pipeline_strings):
            data_tuple = (code, string, 'UNREAD', '-')
            cursor.execute(sql_command, data_tuple)

def sql_logerror(error, pipeline, time):
    """Log errors of the generational process into a DB."""

    def sql_createtable_errors_gen_process(cursor):
        """ Check whether table exists in db and create it if it does not.
        """

        create_table_command = """
        CREATE TABLE errors_gen_process (
        error VARCHAR(180), 
        pipeline VARCHAR(180), 
        times TIMESTAMP);"""

        # get the count of tables with the name
        cursor.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='errors_gen_process' ''')

        # if the count is 1, then table exists
        if cursor.fetchone()[0] == 1:
            # db exists already
            pass
        else:
            cursor.execute(create_table_command)
            print("table 'errors_gen_process' created.")

    with sqlite3.connect(db_path, timeout=120.0) as connection:
        cursor = connection.cursor()
        sql_createtable_errors_gen_process(cursor)
        sql_command = """INSERT INTO errors_gen_process (error, pipeline, times)
        VALUES (?,?,?);"""
        data_tuple = (error, pipeline, time)
        cursor.execute(sql_command, data_tuple)

def sql_writetodb(evaluated_individuals_, result_score_list, eval_individuals_str, duplicates, pipeline_codes):
    """Log results of an evaluated generation in a DB."""

    def sql_createtable_dauersuche(cursor):
        """ Check whether table exists in db and create it if it does not.
        """

        create_table_command = """
        CREATE TABLE dauersuche ( 
        n_iter INTEGER PRIMARY KEY,
        seed INT,
        generation INT,
        id INT, 
        pipeline VARCHAR(180), 
        cv_score FLOAT,
        operator_count INT, 
        t_eval_mins FLOAT,
        pm_cv FLOAT, 
        stator_yoke_cv FLOAT, 
        stator_tooth_cv FLOAT,
        stator_winding_cv FLOAT,
        mutation_count INT, 
        crossover_count INT,
        predecessor VARCHAR(30),
        pipeline_code VARCHAR(180),
        starttime TIMESTAMP);"""

        # get the count of tables with the name
        cursor.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='dauersuche' ''')

        # if the count is 1, then table exists
        if cursor.fetchone()[0] == 1:
            # db exists already
            pass
        else:
            cursor.execute(create_table_command)
            print("Table 'dauersuche' created.")

    with sqlite3.connect(db_path, timeout=120.0) as connection:
        cursor = connection.cursor()
        sql_createtable_dauersuche(cursor=cursor)

        # Duplicates
        sql_command = """INSERT INTO dauersuche (n_iter, seed, generation, id, pipeline, cv_score, operator_count, t_eval_mins,
        pm_cv, stator_yoke_cv, stator_tooth_cv, stator_winding_cv, mutation_count, crossover_count, predecessor,
        pipeline_code, starttime)
        VALUES (NULL,?,?,'-',?,?,'-','-','-','-','-','-','-','-','-','-',?);"""

        for dup in duplicates:
            cursor.execute(sql_command, dup)

        # Evaluated Pipelines
        sql_command = """INSERT INTO dauersuche (n_iter, seed, generation, id, pipeline, cv_score, operator_count, t_eval_mins,
            pm_cv, stator_yoke_cv, stator_tooth_cv, stator_winding_cv, mutation_count, crossover_count, predecessor,
            pipeline_code, starttime)
        VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);"""

        for idx, (result_score, individual_str, code) in enumerate(zip(result_score_list, eval_individuals_str, pipeline_codes)):
            _, pm_cv, stator_yoke_cv, stator_tooth_cv, stator_winding_cv, t_eval, t_start = result_score
            data_tuple = (tpot_settings.random_state,
                          evaluated_individuals_[individual_str]['generation'],
                          idx,
                          individual_str,
                          evaluated_individuals_[individual_str]['internal_cv_score'],
                          evaluated_individuals_[individual_str]['operator_count'],
                          t_eval,
                          pm_cv,
                          stator_yoke_cv,
                          stator_tooth_cv,
                          stator_winding_cv,
                          evaluated_individuals_[individual_str]['mutation_count'],
                          evaluated_individuals_[individual_str]['crossover_count'],
                          str(evaluated_individuals_[individual_str]['predecessor']),
                          code,
                          t_start)

            cursor.execute(sql_command, data_tuple)