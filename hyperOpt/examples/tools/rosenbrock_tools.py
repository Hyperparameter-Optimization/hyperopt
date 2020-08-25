import numpy as np


def rosenbrock_function(
        parameter_dict,
        a=1,
        b=100
):
    ''' The Rosenbrock function

    Parameters:
    -----------
    parameter_dict : dict
        Dictionary containing the hyperparameters (the coordinates of the
        point to be evaluated)
    [a=1] : float
        The parameter 'a' of the Rosenbrock function
    [b=100] : float
        The parameter 'b' of the Roisenbrock function

    Returns:
    -------
    score : float
        The function valueat the coordinates 'x' and 'y'. Returns the negative
        Rosenbrock function value.
    '''
    score = (
        (a - parameter_dict['x'])**2
        + b*(parameter_dict['y']- parameter_dict['x']**2)**2
    )
    return (-1)*score


def ensemble_rosenbrock(
        parameter_dicts,
        true_values={'a': 1, 'b': 100}
):
    ''' Calcualtes the Rosenbrock function value for the ensemble.

    Parameters:
    -----------
    parameter_dicts : list of dicts
        List of the coordinate dictionaries
    true_values : dict
        Values for the 'a' and 'b' parameter for the Rosenbrock function

    Returns:
    --------
    scores : list
        Scores for each member in the ensemble
    '''
    scores = []
    for parameter_dict in parameter_dicts:
        score = rosenbrock_function(
            parameter_dict,
            true_values['a'],
            true_values['b'])
        scores.append(score)
    return scores


def check_distance(true_values, best_parameters):
    '''Calculates the distance of the parameters from the true location
    of the minima

    Parameters:
    ----------
    true_values : dict
        Values for the 'a' and 'b' parameter for the Rosenbrock function
    best_parameters : dict
        The 'x' and 'y' values of the best found location

    Returns:
    -------
    distance  float
        Eucledian distance of the parameters from the true minima of the
        Rosenbrock function
    '''
    true_parameters = {'x': true_values['a'], 'y': true_values['a']**2}
    diff_dict = {}
    diff_sqared_sum = 0
    for key in true_parameters:
        diff_dict[key] = true_parameters[key] - best_parameters[key]
        diff_sqared_sum += diff_dict[key]**2
    distance = np.sqrt(diff_sqared_sum)
    return distance


def run_random(
        parameter_dicts,
        value_dicts,
        settings
):
    '''Performs the whole particle swarm optimization

    Parameters:
    ----------
    global_settings : dict
        Global settings for the run.
    settings : dict
        Settings for the run
    parameter_dicts : list of dicts
        The parameter-sets of all particles.

    Returns:
    -------
    best_parameters : dict
        dictionary of optimal location
    best_fitness : float
        Value of the best location
    '''
    print(':::::::: Initializing :::::::::')
    iterations = settings['iterations']
    true_values = {'a': settings['a'], 'b': settings['b']}
    i = 1
    new_parameters = parameter_dicts
    personal_bests = {}
    fitnesses = ensemble_rosenbrock(parameter_dicts, true_values)
    index = np.argmax(fitnesses)
    best_fitness = fitnesses[index]
    best_parameters = parameter_dicts[index]
    personal_bests = parameter_dicts
    best_fitnesses = fitnesses
    print('::::::::::: Optimizing ::::::::::')
    while i <= iterations:
        print('---- Iteration: ' + str(i) + '----')
        parameter_dicts = prepare_run_params(
            value_dicts, settings['sample_size'])
        fitnesses = ensemble_rosenbrock(parameter_dicts, true_values)
        if best_fitness > max(fitnesses):
            index = np.argmax(fitnesses)
            best_fitness = max(fitnesses)
            best_parameters = parameter_dicts[index]
        distance = check_distance(true_values, best_parameters)
        i += 1
    return best_parameters, best_fitness


def prepare_run_params(value_dicts, sample_size):
    ''' Creates parameter-sets for all particles (sample_size)

    Parameters:
    ----------
    value_dicts : list of dicts
        Specifications how each value should be initialized
    sample_size : int
        Number of particles to be created

    Returns:
    -------
    run_params : list of dicts
        List of parameter-sets for all particles
    '''
    run_params = []
    for i in range(sample_size):
        run_param = initialize_values(value_dicts)
        run_params.append(run_param)
    return run_params


def initialize_values(value_dicts):
    '''Initializes the parameters according to the value dict specifications

    Parameters:
    ----------
    value_dicts : list of dicts
        Specifications how each value should be initialized

    Returns:
    -------
    sample : list of dicts
        Parameter-set for a particle
    '''
    sample = {}
    for parameters in value_dicts:
         sample[str(parameters['parameter'])] = np.random.uniform(
            low=parameters['min'],
            high=parameters['max']
        )
    return sample


def flatten_dicts(result_dicts, true_values):
    '''Creates a dict of lists from list of dicts.

    Parameters:
    ----------
    result_dicts : list of dicts
        List of dictionaries containing 'best_parameters' value
    true_values : dict
        Values for the 'a' and 'b' parameter for the Rosenbrock function

    Returns:
    -------
    methods_dict : dict of dicts
        Dictionary containing dictionaries for each method. Each method
        dictionary contains the best fitnesses and best distances.
    '''
    keys = list(result_dicts[0].keys())
    methods_dict = {}
    for key in keys:
        method_dict = {
            'fitnesses': [],
            'distances': []
        }
        for result_dict in result_dicts:
            best_parameters = result_dict[key]['best_parameters']
            best_fitness = result_dict[key]['best_fitness']
            best_distance = check_distance(true_values, best_parameters)
            method_dict['fitnesses'].append(best_fitness)
            method_dict['distances'].append(best_distance)
        methods_dict[key] = method_dict
    return methods_dict