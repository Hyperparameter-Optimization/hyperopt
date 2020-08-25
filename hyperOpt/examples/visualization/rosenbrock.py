'''Visualization tools for the Rosenbrock function'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import docopt
import json
import matplotlib.ticker as ticker
import numpy as np
import os


def plot_absolute_distances(absolute_distances, rnd, label, bins, hatch=None):
    label_dict = {
        'ga': 'GA',
        'pso': 'PSO',
        'gd': 'GD',
        'random': 'RD',
        'gs': 'GR'
    }
    bins = plt.hist(
        absolute_distances,
        alpha=0.3,
        bins=bins,
        label=label_dict[label],
        hatch=hatch,
        ec="k",
        histtype='stepfilled'
    )[1]
    if rnd:
        plt.title("Absolute distance from minimum")
        plt.xlabel("Distance")
        plt.ylabel("# cases")
    return bins


def plot_fitness_values(best_fitnesses_list, rnd, label, bins, hatch=None):
    label_dict = {
        'ga': 'GA',
        'pso': 'PSO',
        'gd': 'GD',
        'random': 'RD',
        'gs': 'GR'
    }
    bins = plt.hist(
        best_fitnesses_list,
        alpha=0.3,
        bins=bins,
        label=label_dict[label],
        hatch=hatch,
        ec="k",
        histtype='stepfilled',
        lw=1.0
    )[1]
    font = {
        'size': 12
    }
    if rnd:
        plt.title("Fitness values")
        plt.xlabel(r"$\hat{\hat{s}}$", fontdict=font)
        plt.ylabel("Runs per bin", fontdict=font)
    return bins


def plot_fitnesses_history(result_dict, rnd, label):
    old_fitnesses = result_dict['list_of_best_fitnesses']
    x_values = np.arange(len(old_fitnesses))
    plt.plot(
        x_values,
        old_fitnesses,
        label=label
    )
    if rnd:
        plt.xlabel('Iteration number / #')
        plt.ylabel('Fitness')


def plot_distances(result_dict, rnd, label):
    best_parameters_list = result_dict['list_of_old_bests']
    x_distances = np.array([np.abs(i['x'] - 1) for i in best_parameters_list])
    y_distances = np.array([np.abs(i['y'] - 1) for i in best_parameters_list])
    absolute_distances = np.sqrt(x_distances**2 + y_distances**2)
    x_values = np.arange(len(absolute_distances))
    plt.plot(
        x_values,
        absolute_distances,
        label=label)
    if rnd:
        plt.xlabel('Iteration number / #')
        plt.ylabel('Distance')


def read_stability_files(output_dir, methods='all'):
    input_dir = os.path.join(output_dir)
    method_list = ['ga', 'gd', 'pso', 'random']
    method_dicts = {}
    if methods == 'all':
        for method in method_list:
            method_dicts = load_one_stability_file(
                method, input_dir, method_dicts)
    else:
        for method in methods:
            method_dicts = load_one_stability_file(
                method, input_dir, method_dicts)
    return method_dicts


def load_one_stability_file(method, input_dir, method_dicts):
    method_dicts[method] = {}
    result_path = os.path.join(input_dir, method + '_result.json')
    with open(result_path, 'rt') as infile:
        for line in infile:
            method_dicts[method].update(json.loads(line))
    return method_dicts


def plot_loaded_stability_main(method_dicts, output_dir):
    new_out = os.path.join(output_dir, 'stabilities')
    plot_loaded_stabilities(method_dicts, new_out, 'fitnesses')
    plot_loaded_stabilities(method_dicts, new_out, 'distances')
    find_std_mean(method_dicts, new_out)


def plot_loaded_stabilities(method_dicts, new_out, to_plot):
    if not os.path.exists(new_out):
        os.makedirs(new_out)
    keys = list(method_dicts.keys())
    if to_plot == 'fitnesses':
        bins = np.logspace(-7, 3, 30)
    else:
        bins = np.logspace(-4, np.log10(np.sqrt(2)*500), 15)
    hatches = ['////', '|||', '---', '\\\\\\\\', '....']
    for i, key in enumerate(keys):
        if key == 'random':
            rnd = True
        else:
            rnd = False
        if to_plot == 'distances':
            method_distances = method_dicts[key]['distances']
            bins = plot_absolute_distances(
                method_distances, rnd, key, bins, hatch=hatches[i])
        else:
            method_fitnesses = method_dicts[key]['fitnesses']
            bins = plot_fitness_values(
                method_fitnesses, rnd, key, bins, hatch=hatches[i])
    plt.legend()
    plt.xscale('log')
    output_path = os.path.join(new_out, 'best_' + to_plot + '.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')


def find_std_mean(method_dicts, output_dir):
    method_list = method_dicts.keys()
    for method in method_list:
        fitnesses = method_dicts[method]['fitnesses']
        mean = np.mean(fitnesses)
        stdev = np.std(fitnesses)
        out_dict = {"method": method, "mean": mean, "stdev": stdev}
        output_path = os.path.join(output_dir, 'stability_results.json')
        with open(output_path, 'at') as out_file:
            json.dump(out_dict, out_file)
            out_file.write('\n')
