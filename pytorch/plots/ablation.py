from matplotlib import pyplot as plt
from os import path as osp
from itertools import zip_longest, product
import numpy as np
import argparse, sys, os
import pandas as pd
from scipy.ndimage import uniform_filter
import matplotlib as mpl
mpl.style.use('seaborn')

def unpack(s):
    return " ".join(map(str, s))

def remove_nan(raw_data):
    return raw_data[~np.isnan(raw_data)]

args = argparse.ArgumentParser(sys.argv[0])
args.add_argument('--dir', type = str)
args = args.parse_args()
if args.dir[-1] == '/':
    args.dir = args.dir[:-1]
parts = args.dir.split('/')
save_name = parts[-2] + '/' + parts[-1]
print(save_name)

divs = sorted(os.listdir(args.dir))

metrics = ['Train Loss', 'Train Acc', 'Test Loss', 'Test Acc']

x_axis = 'Epoch'

for div in divs:
    fig = plt.figure(figsize=(7*2, 5*2))
    axes = [fig.add_subplot(2,2,i+1) for i in range(4)]
    for trial_id, trial in enumerate(sorted(os.listdir(osp.join(args.dir, div)))):
        file_path = osp.join(args.dir, div, trial, 'progress.csv')
        print(file_path)
        if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
            continue
        df = pd.read_csv(file_path)

        y = remove_nan(df.loc[:, x_axis].values)

        for metric, ax in zip(metrics, axes):
            if metric not in list(df):
                continue
            raw_data = remove_nan(df.loc[:, metric].values) # to numpy array
            data = uniform_filter(raw_data, 20)
            ax.plot(y[:len(data)], data, label=f'{div}_{trial_id}')
            ax.text(y[-1], data[-1], f'{trial_id}')
    
    for metric, ax in zip(metrics, axes):
        ax.set_xlabel(x_axis)
        ax.set_ylabel(metric)
        # ax.legend()
    
    plt.suptitle(f"{save_name}-{div}", y=0.98)
    plt.tight_layout()
    os.makedirs(f'imgs/{save_name}/', exist_ok=True)
    plt.savefig(f'imgs/{save_name}/{div}.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close()
