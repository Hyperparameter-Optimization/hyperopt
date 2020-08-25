'''
Call with 'python'

Usage: higgs_ga_xgb_pso.py --data_path=PTH --use_slurm=BOOL

Options:
    --data_path=PTH              Path to parameters to be run
    --use_slurm=INT              Whether to use slurm or computate locally

'''
from hyperOpt.examples.tools import submission_higgs as sh
from hyperOpt.examples.tools import slurm_tools as st
from hyperOpt.examples.tools import xgb_tools as xt
from hyperOpt.tools import genetic_algorithm as ga
from hyperOpt.tools import universal_tools as ut
import numpy as np
import docopt
import json
import os

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
        use_slurm,
        global_settings,
        settings_dir,
        aux_settings_dir
        train_file='training.csv'
):
    print("::::::: Reading parameters :::::::")
    param_file = os.path.join(
        aux_settings_dir,
        'xgb_parameters.json'
    )
    parameter_infos = ut.read_parameters(param_file)
    ga_settings = ut.read_settings(settings_dir, 'ga')
    ga_settings.update(global_settings)
    if use_slurm:
        evaluation = st.get_fitness_score
    else:
        evaluation = xt.ensemble_fitness
    population = Population(parameter_infos, evaluation, ga_settings)
    optimal_hyperparameters, best_fitness = swarm.survivalOfTheFittest()
    return optimal_hyperparameters


def save_results(optimal_hyperparameters, data_path, output_dir):
    path_to_test = os.path.join(data_path, 'test.csv')
    path_to_train = os.path.join(data_path, 'training.csv')
    score_path = os.path.join(output_dir, 'best_hyperparameters.json')
    outfile = os.path.join(output_dir, 'higgs_submission.ga')
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
    print('GA submission file: ' + str(outfile))


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        data_path = arguments['--data_path']
        use_slurm = bool(int(arguments['--use_slurm']))
        main(data_path, use_slurm)
    except docopt.DocoptExit as e:
        print(e)
    result_dict, output_dir = main()
