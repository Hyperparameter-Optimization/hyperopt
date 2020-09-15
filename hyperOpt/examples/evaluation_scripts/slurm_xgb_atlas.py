'''
Call with 'python3'

Usage: slurm_xgb_atlas.py --parameter_file=PTH --output_dir=DIR

Options:
    -p --parameter_file=PTH      Path to parameters to be run
    --output_dir=DIR             Directory of the output


'''
import numpy as np
from hyperOpt.examples.tools import universal_tools as ut
from hyperOpt.examples.tools import atlas_tools as at
from hyperOpt.examples.tools import xgb_tools as xt
import docopt
import json
from pathlib import Path
import os
np.random.seed(1)

path_to_file = os.path.expandvars("$HOME/training.csv")


def main(parameter_file, output_dir):
    settings_dir = os.path.join(output_dir, 'run_settings')
    global_settings = ut.read_settings(settings_dir, 'global')
    num_classes = global_settings['num_classes']
    nthread = global_settings['nthread']
    parameter_dict = ut.read_parameters(parameter_file)[0]
    path = Path(parameter_file)
    save_dir = str(path.parent)
    d_amss, test_amss, train_amss = at.higgs_evaluation_main(
        path_to_file, parameter_dict)
    print('d_ams scores: ' + str(d_amss))
    print('test_AMS scores: ' + str(test_amss))
    print('train_AMS scores: ' + str(train_amss))
    mean_d_amss = np.mean(d_amss)
    std_d_amss = np.std(d_amss)
    score = mean_d_amss - std_d_amss
    score_path = os.path.join(save_dir, 'score.json')
    with open(score_path, 'w') as score_file:
        json.dump(
            {
                'd_ams': score,
                'test_ams': np.mean(test_amss),
                'std_test_ams': np.std(test_amss),
                'mean_d_ams': mean_d_amss,
                'std_d_ams': std_d_amss
            },
            score_file
        )


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        parameter_file = arguments['--parameter_file']
        output_dir = arguments['--output_dir']
        main(parameter_file, output_dir)
    except docopt.DocoptExit as e:
        print(e)