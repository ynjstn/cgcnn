import subprocess
from itertools import product
import pandas as pd

'''
Param list
- epochs
- batch size
- learning rate
- milestone
- momentum
- weight-decay
- train ratio
- val ratio
- test ratio
- optim SGD
- atom fea len
- h fea len
- n-conv
- n-h
'''


grid_result = pd.DataFrame(columns=['MAE', 'epochs', 'batch-size', 'learning_rate', 'optim_SGD', 'atom_fea_len', 'h_fea_len', 'n_conv', 'n_h'])
grid_error = pd.DataFrame(columns=['epochs', 'batch-size', 'learning_rate', 'optim_SGD', 'atom_fea_len', 'h_fea_len', 'n_conv', 'n_h'])

param_grid = {
    'epochs': [2],
    'batch-size': [103, 206, 256],
    'learning-rate': [0.001, 0.05, 0.01, 0.1],
    'optim': ['SGD', 'Adam'],
    'atom-fea-len': [16, 31, 64, 128],
    'h-fea-len': [32, 64, 128, 256],
    'n-conv': [1, 3, 5, 8],
    'n-h': [1, 3, 5, 8]
}


def main():

    global grid_result, grid_error, param_grid

    checkpoint = 0
    nbr_errors = 0
    combination = list(product(*param_grid.values()))
    for param in combination:
        cmd = commande_line(param)
        checkpoint, nbr_errors = run_gridsearch(cmd, param, checkpoint, nbr_errors)
        print(grid_result.head())

    grid_result.to_csv('Gridsearch/gridsearch_result.csv')
    print('Gridsearch finished and saved')
    print(f'Number of errors: {nbr_errors}')


def commande_line(param):
    cmd = ['python', 'main.py',
           '--train-ratio', '0.6',
           '--test-ratio', '0.2',
           '--val-ratio', '0.2',
           '--gridsearch',
           '--epochs', str(param[0]),
            '--epochs', str(param[0]),
            '--batch-size', str(param[1]),
            '--lr', str(param[2]),
            '--optim', param[3],
            '--atom-fea-len', str(param[4]),
            '--h-fea-len', str(param[5]),
            '--n-conv', str(param[6]),
            '--n-h', str(param[7]),
            'data/Cubic_lattice']
    return cmd


def run_gridsearch(cmd, param, checkpoint, nbr_errors):
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, check=True)
        print(f'Last MAE : {result.stdout}', param)
        grid_result.loc[len(grid_result)] = [result.stdout, param[0], param[1], param[2], param[3], param[4], param[5], param[6], param[7]]
        checkpoint += 1
        print(checkpoint)
    except subprocess.CalledProcessError as e:
        print('Error, iteration passed. Hyperparameters added to grid_error')
        grid_error.loc[len(grid_error)] = [param[0], param[1], param[2], param[3], param[4], param[5], param[6], param[7]]
        grid_error.to_csv('Gridsearch/gridsearch_errors.csv')
        nbr_errors += 1

    if checkpoint % 2 == 0:
        print('toto')
        grid_result.to_csv('Gridsearch/gridsearch_result.csv')
        print(f'Result save to csv file (iteration {checkpoint})')

    return checkpoint, nbr_errors


if __name__ == '__main__':
    main()
