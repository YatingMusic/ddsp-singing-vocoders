import os
import json
import time
import torch
import logging
import datetime
import collections
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal


def load_loss(path_log):
    monitor_vals = collections.defaultdict(list)
    with open(path_log, 'r') as f:
        for line in f:
            try:
                line = line.strip()
                key, val, step, acc_time = line.split(' | ')
                monitor_vals[key].append((float(val), int(step), acc_time))
            except:
                continue
    data_val = dict()
    data_tra = dict()
    data_val['vals'] = [item[0] for item in monitor_vals['valid loss']]
    data_val['step'] = [item[1] for item in monitor_vals['valid loss']]
    data_val['time'] = [item[2] for item in monitor_vals['valid loss']]
    
    data_tra['vals'] = [item[0] for item in monitor_vals['train loss']]
    data_tra['step'] = [item[1] for item in monitor_vals['train loss']]
    data_tra['time'] = [item[2] for item in monitor_vals['train loss']]
    
    return data_val, data_tra
    
def make_loss_report(
        exp_list,
        title,
        path_fig):
    fig = plt.figure(dpi=150)
    plt.title(title)
#     colors = ['b', 'r', 'g', 'black']
    for idx, (exp, exp_label) in enumerate(exp_list):
#         c = colors[idx]
        path_log = os.path.join(exp, 'log_value.txt')
        data_val, data_tra = load_loss(path_log)
        plt.plot(data_val['step'], data_val['vals'], label=exp_label, linestyle='-')
#         plt.plot(data_tra['step'], data_tra['vals'], label=exp+' tra', linestyle=':', c=c)
        
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.tight_layout()
#     plt.xlim([0, 160000])
    # plt.ylim([0.001, 0.5])
    plt.savefig(path_fig)

if __name__ == '__main__':
    exp_list = [
        ('./exp/f1-full/sins',          'sins-new'),
        ('./exp/f1-full/sawsinsub-256', 'sawsinsub-256'),
    ]
    make_loss_report(exp_list, 'SVS', 'exp/compare.png')