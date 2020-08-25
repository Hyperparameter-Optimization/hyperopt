'''Tools for grid search.
'''
from itertools import product
import numpy as np
from hyperOpt.examples.tools import rosenbrock_tools as rt

def create_all_combinations(nr_parameters, number_cuts):
    '''Creates all possible combinations of parameters

    Parameters:
    ----------
    nr_parameters : int
        Number of parameters that are in the grid
    number_cuts : int
        Number of values to be added to the grid per parameter

    Returns:
    -------
    combinations : list
        List of all the combinations
    '''
    combinations = list(product(
        range(number_cuts),
        repeat=nr_parameters))
    return combinations


def initialize_values(parameters, number_cuts, random_nudge):
    '''Initializes the values of the grid

    Parameters:
    ----------

    parameters : list of dicts
        Each element of the list contains the necessary info for each
        parameter
    number_cuts : int
        Number of values in the grid per parameter
    random_nudge : float
        Randomoffset of the grid in a given direction

    Returns:
    -------

    parameter_dicts : list of dicts
        List of dictionaries that constitute the whole grid and that
        cover the whole grid.
    '''
    parameter_dicts = []
    combinations = create_all_combinations(len(parameters), number_cuts)
    for iterations in combinations:
        parameter_dict = single_paramset(
            parameters, iterations, number_cuts, random_nudge)
        parameter_dicts.append(parameter_dict)
    return parameter_dicts


def single_paramset(parameters, iterations, number_cuts, random_nudge):
    '''Creates parameter dict with appropriate parameters

    Parameters:
    ----------
    parameters : list of dicts
        Each element of the list contains the necessary info for each
        parameter
    iterations: list
        Combination
    number_cuts : int
        Number of values in the grid per parameter
    '''
    parameter_dict = {}
    for param, iteration in zip(parameters, iterations):
        key = param['parameter']
        range_size = param['max'] - param['min']
        if number_cuts == 1:
            value = (param['max'] + param['min']) / 2
        else:
            step_size = range_size / (number_cuts - 1)
            value = param['min'] + (iteration * step_size)
        parameter_dict[key] = value + random_nudge[key]
    return parameter_dict


def perform_gridsearch(
        parameters,
        settings,
):
    '''Performs grid search with given settings and parameters.

    Parameters:
    ----------
    parameters : list of dicts
        Each element of the list contains the necessary info for each
        parameter
    data_dict : dict
        Dictionary that contains data and the correct labels
    settings: dict
        Dictionary that contains the necessary parameters for grid search

    Returns:
    -------
    result_dict : dict
        Dictionary that contains the data_dict, pred_train, pred_test and
        the best_parameters
    '''
    true_values = {'a': settings['a'], 'b': settings['b']}
    random_nudge = {
        'x': np.random.uniform(low=-0.5, high=0.5),
        'y': np.random.uniform(low=-0.5, high=0.5)
    }
    number_cuts = calculate_grid_size(settings)
    parameter_dicts = initialize_values(
        parameters, number_cuts, random_nudge
    )
    print(':::::: Calculating fitnesses ::::::')
    fitnesses = rt.ensemble_rosenbrock(parameter_dicts, true_values)
    index = np.argmax(fitnesses)
    best_parameters = parameter_dicts[index]
    best_distance = rt.check_distance(true_values, best_parameters)
    result_dict = {
        'best_parameters': parameter_dicts[index],
        'best_fitness': fitnesses[index],
        'best_distance': best_distance,
    }
    return result_dict


def calculate_grid_size(settings):
    ''' Finds number of cuts in each parameter

    Parameters:
    ----------
    settings : dict
        Settings containing the keys sample_size and iterations

    Returns:
    -------
    number_cuts : int
        Number of cuts in each parameter for the grid
    '''
    number_evaluations = settings['sample_size'] * settings['iterations']
    number_cuts = int(np.sqrt(number_evaluations))
    return number_cuts