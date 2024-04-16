from subprocess import call
from os.path import join
from sql_utilities import ROOT_PATH
from settings.tpot_settings import res_plan


def build_shell_script_lines(pipe_idx, gen_id, tpot_seed):
    lines = ['#!/usr/bin/sh', '',
             '#CCS -t ' + res_plan['duration'],
             f'#CCS -o pc2/logs/job-%reqid-seed-{tpot_seed}-gen-{gen_id}-id-{pipe_idx}.out',
             f'#CCS --stderr=pc2/logs/job-%reqid-seed-{tpot_seed}-gen-{gen_id}-id-{pipe_idx}.err',
             '#CCS --res=rset=' + res_plan['rset'] +
             ':ncpus=' + res_plan['ncpus'] +
             ':mem=' + res_plan['mem'] +
             ':vmem=' + res_plan['vmem'],
             'idx=' + str(pipe_idx),
             "python pipeline_evaluation.py $idx"]
    return [line + '\n' for line in lines]


def create_n_run_script(name, content, dry=False):
        file = join(ROOT_PATH, 'pc2', 'jobs', name) + '.sh'
        with open(file, 'w+') as f:
            f.writelines(content)
        call(["chmod", "+x", file])  # Make script executable
        if not dry:
            call(['ccsalloc', file])  # allocate and run