from hyperOpt.examples.tools import submission_higgs as sh
from hyperOpt.examples.tools import universal_tools as ut
from hyperOpt.examples.tools import slurm_tools as st
from hyperOpt.examples.tools import atlas_tools as at
from hyperOpt.examples.tools import xgb_tools as xt
from hyperOpt import examples as ex
from pathlib import Path
import hyperOpt as ho
import numpy as np
import os
import json
import shutil


NR_EVALUATION = 2
DATA_PATH = '/home/laurits'

def higgs_random():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    current_dir = Path(dir_path)
    package_dir = str(current_dir.parent)
    settings_dir = os.path.join(ho.__path__[0], 'settings')
    aux_settings_dir = os.path.join(ex.__path__[0], 'settings')
    global_settings = ut.read_settings(aux_settings_dir, 'global')
    global_settings['package_dir'] = str(package_dir)
    global_settings['output_dir'] = str(os.path.expandvars(
        global_settings['output_dir']))
    output_dir = os.path.expandvars(global_settings['output_dir'])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
    ut.save_run_settings(output_dir, settings_dir, aux_settings_dir)
    param_file = os.path.join(
        aux_settings_dir,
        'xgb_parameters.json'
    )
    parameter_infos = ut.read_parameters(param_file)
    parameters = xt.prepare_run_params(parameter_infos, NR_EVALUATION)
    fitnesses = st.get_fitness_score(parameters, global_settings)
    index = np.argmax(fitnesses)
    best_parameters = parameters[index]
    best_param_path = os.path.join(
        global_settings['output_dir'], 'best_parameters.json')
    with open(best_param_path, 'wt') as outFile:
        json.dump(best_parameters, outFile)
    path_to_test = os.path.join(DATA_PATH, 'test.csv')
    path_to_train = os.path.join(DATA_PATH, 'training.csv')
    submission_file = os.path.join(
        global_settings['output_dir'], 'higgs_submission.rand')
    print('Creating submission file')
    sh.submission_creation(
        path_to_train,
        path_to_test,
        best_parameters,
        outfile
    )

if __name__ == '__main__':
    higgs_random()