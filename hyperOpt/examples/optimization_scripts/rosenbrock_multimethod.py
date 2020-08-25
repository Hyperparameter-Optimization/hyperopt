'''Run stability and/or performance for different methods of finding a minima
of the Rosenbrock function.

Call with 'python'

Usage: rosenbrock_multitool.py  --output_dir=DIR --method=STR

Options:

    --output_dir=DIR         Directory of the output
    --choice=STR             Either to run 'stability', 'performance' or 'both'
    --method=STR             Either to run 'ga', 'pso', 'gd', 'all'
'''
from hyperOpt.examples.visualization import rosenbrock as rplt
from hyperOpt.examples.tools import grid_search_tools as gst
from hyperOpt.examples.tools import rosenbrock_tools as rt
from hyperOpt.examples.tools import universal_tools as ut
from hyperOpt.examples.tools import gradient_tools as gd
from hyperOpt.tools import genetic_algorithm as ga
from hyperOpt.tools import particle_swarm as ps
from hyperOpt import examples as ex
import hyperOpt as ho
from pathlib import Path
import docopt
import json
import numpy as np
import os


def main(method, output_dir, true_values={'a': 1, 'b': 100}):
    print('::::::: Reading settings and parameters :::::::')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(ho.__path__)
    settings_dir = os.path.join(ho.__path__[0], 'settings')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    all_settings = read_all_settings(settings_dir, true_values)
    param_file = os.path.join(
        ex.__path__[0], 'settings', 'rosenbrock_parameters.json')
    value_dicts = ut.read_parameters(param_file)
    parameter_dicts = rt.prepare_run_params(
        value_dicts, all_settings['pso']['sample_size'])
    i = 0
    result_dicts = []
    print("Testing stability")
    while i < 100:
        np.random.seed(i)
        result_dict = run_single_choice(
            all_settings,
            parameter_dicts,
            value_dicts,
            method
        )
        result_dicts.append(result_dict)
        i += 1
    methods_dict = rt.flatten_dicts(result_dicts, true_values)
    rplt.plot_loaded_stability_main(methods_dict, output_dir)
    stability_info_dir = os.path.join(output_dir, 'stabilities')
    save_stability_info(result_dicts, stability_info_dir)


def read_all_settings(settings_dir, true_values):
    ''' Reads the settings for all different methods

    Parameters:
    ----------
    settings_dir : str
        Path to the directory where all the settings are stored

    Returns:
    -------
    all_settings : dict
        Dictionary containing the settings for the Genetic algorithm,
        Particle Swarm Optimization, Gradient Descent, Random Guessing and
        Grid Search
    '''
    all_settings = {}
    methods = ['ga', 'pso']
    for method in methods:
        all_settings[method] = ut.read_settings(settings_dir, method)
        all_settings[method].update(true_values)
    all_settings['gd'] = ut.read_settings(
        os.path.join(ex.__path__[0], 'settings'), 'gd'
    )
    all_settings['gd'].update(true_values)
    iterations = all_settings['ga']['iterations']
    sample_size = all_settings['ga']['sample_size']
    all_settings['random'] = {
        'iterations': iterations,
        'sample_size': sample_size
    }
    all_settings['random'].update(true_values)
    all_settings['gs'] = all_settings['random'].copy()
    return all_settings


def run_single_choice(
        all_settings,
        parameter_dicts,
        value_dicts,
        method
):
    result_dict = {}
    print("Random choice")
    rnd_best_parameters, rnd_best_fitness = rt.run_random(
        parameter_dicts,
        value_dicts,
        all_settings['random']
    )
    result_dict['random'] = {
        'best_fitness': (-1)*rnd_best_fitness,
        'best_parameters': rnd_best_parameters
    }
    if method == 'ga' or method == 'all':
        print("Genetic Algorithm")
        population = ga.Population(
            value_dicts, rt.ensemble_rosenbrock, all_settings['ga'])
        best_parameters, best_fitness = population.survivalOfTheFittest()
        result_dict['ga'] ={
            'best_parameters': (-1)*best_parameters,
            'best_fitness': best_fitness
        }
    if method == 'pso' or method == 'all':
        print("Particle swarm optimization")
        swarm = ps.ParticleSwarm(
            all_settings['pso'], rt.ensemble_rosenbrock, value_dicts)
        pso_best_parameters, pso_best_fitness = swarm.particleSwarmOptimization()
        result_dict['pso'] = {
            'best_fitness': (-1)*pso_best_fitness,
            'best_parameters': pso_best_parameters
        }
    if method == 'gd' or method == 'all':
        print("Gradient descent")
        result_dict['gd'] = gd.gradient_descent(
            all_settings['gd'],
            value_dicts,
            rt.initialize_values,
            rt.rosenbrock_function,
            rt.check_distance
        )
    if method == 'gs' or method == 'all':
        print('Grid search')
        result_dict['gs'] = gst.perform_gridsearch(
            value_dicts,
            all_settings['gs'],
        )
        result_dict
    return result_dict


def save_stability_info(result_dicts, output_dir):
    new_out = os.path.join(output_dir, 'collected_data')
    if not os.path.exists(new_out):
        os.makedirs(new_out)
    keys = result_dicts[0].keys()
    for key in keys:
        distances = []
        fitnesses = []
        for result_dict in result_dicts:
            method_dict = result_dict[key]
            x = method_dict['best_parameters']['x']
            y = method_dict['best_parameters']['y']
            dist = np.sqrt(np.abs(x - 1)**2 + np.abs(y - 1)**2)
            distances.append(dist)
            fitnesses.append(method_dict['best_fitness'])
        output_path = os.path.join(new_out, key + '.json')
        with open(output_path, 'w') as out_file:
            json.dump({"distances": distances}, out_file)
            out_file.write('\n')
            json.dump({"fitnesses": fitnesses}, out_file)


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        method = arguments['--method']
        output_dir = arguments['--output_dir']
        main(method, output_dir)
    except docopt.DocoptExit as e:
        print(e)