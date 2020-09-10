'''
Call with 'python'

Usage: higgs_pso_xgb_pso.py --data_path=PTH --use_slurm=BOOL

Options:
    --data_path=PTH              Path to parameters to be run
    --use_slurm=INT              Whether to use slurm or computate locally

'''
from hyperOpt.examples.tools import submission_higgs as sh
from hyperOpt.examples.tools import universal_tools as ut
from hyperOpt.examples.tools import slurm_tools as st
from hyperOpt.examples.tools import atlas_tools as at
from hyperOpt.tools import particle_swarm as pm
from hyperOpt import examples as ex
from pathlib import Path
import hyperOpt as ho
import numpy as np
import shutil
import os
import json
import docopt

np.random.seed(1)


def main(data_path, use_slurm):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    current_dir = Path(dir_path)
    package_dir = str(current_dir.parent)
    settings_dir = os.path.join(ho.__path__[0], 'settings')
    aux_settings_dir = os.path.join(ex.__path__[0], 'settings')
    global_settings = ut.read_settings(aux_settings_dir, 'global')
    global_settings['package_dir'] = str(package_dir)
    global_settings['output_dir'] = str(global_settings['output_dir'])
    output_dir = os.path.expandvars(global_settings['output_dir'])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
    ut.save_run_settings(output_dir, settings_dir, aux_settings_dir)
    result_dict = optimize(
        data_path,
        use_slurm,
        global_settings,
        settings_dir,
        aux_settings_dir
    )
    save_results(result_dict, data_path, output_dir)


def optimize(
        data_path,
        use_slurm,
        global_settings,
        settings_dir,
        aux_settings_dir,
        train_file='training.csv'
):
    print("::::::: Reading parameters :::::::")
    param_file = os.path.join(
        aux_settings_dir,
        'xgb_parameters.json'
    )
    training_file = os.path.join(data_path, train_file)
    value_dicts = ut.read_parameters(param_file)
    pso_settings = ut.read_settings(settings_dir, 'pso')
    pso_settings.update(global_settings)
    pso_settings['train_file'] = training_file
    if use_slurm:
        evaluation = st.get_fitness_score
    else:
        evaluation = at.ensemble_fitness
    swarm = pm.ParticleSwarm(pso_settings, evaluation, value_dicts)
    optimal_hyperparameters = swarm.particleSwarmOptimization()[0]
    return optimal_hyperparameters


def save_results(optimal_hyperparameters, data_path, output_dir):
    path_to_test = os.path.join(data_path, 'test.csv')
    path_to_train = os.path.join(data_path, 'training.csv')
    score_path = os.path.join(output_dir, 'best_hyperparameters.json')
    outfile = os.path.join(output_dir, 'higgs_submission.pso')
    with open(score_path, 'w') as file:
        json.dump(optimal_hyperparameters, file)
    print('Creating submission file')
    sh.submission_creation(
        path_to_train,
        path_to_test,
        optimal_hyperparameters,
        outfile
    )
    print('Results saved to ' + str(output_dir))
    print('PSO submission file: ' + str(outfile))


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        data_path = arguments['--data_path']
        use_slurm = bool(int(arguments['--use_slurm']))
        main(data_path, use_slurm)
    except docopt.DocoptExit as e:
        print(e)

