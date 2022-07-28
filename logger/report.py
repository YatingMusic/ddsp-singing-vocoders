import os
import datetime
import collections
import numpy as np
import matplotlib.pyplot as plt



def load_log(path_log):
    vals_map = collections.defaultdict(list)
    with open(path_log, 'r') as f:
        for line in f:
            try:
                line = line.strip()
                key, val, step, acc_time = line.split(' | ')
                vals_map[key].append((float(val), int(step), acc_time))
            except:
                continue
    return vals_map


def unpack_values(value_list):
    xs, ys, ts = [], [], []
    for val in value_list:
        xs.append(val[1]) 
        ys.append(val[0]) 
        ts.append(val[2])
    return xs, ys, ts


def compare_exp(
        log_list,
        x_range=None,
        y_range=None,
        path_fig='compare.png',
        value='valid loss',
        title='valid loss comparison',
        dpi=120,
        y_log=True):
    
    '''
    several exps, one value
    '''
    print(f'[*] generating report for {value}')
    fig = plt.figure(dpi=dpi)
    plt.title(title)

    exported_data = dict()
    for path_log, name in log_list:
        vals_map = load_log(os.path.join(path_log, 'log_value.txt'))
        xs, ys, ts = unpack_values(vals_map[value])

        xs = np.arange(len(xs))
        plt.plot(xs, ys, label=name)

        min_y = np.min(ys)
        min_x = xs[np.argmin(ys)]
        min_t = ts[np.argmin(ys)]
        plt.plot(min_x, min_y, 'ro')
    
        exported_data[path_log+'_x'] = xs
        exported_data[path_log+'_y'] = ys
        
    if y_log:
        plt.yscale('log')
    plt.legend(loc='upper right')
    if x_range:
        plt.xlim(x_range)
    if y_range:
        plt.ylim(y_range)
    plt.tight_layout()
    plt.savefig(path_fig)
    plt.close() 

    np.savez(path_fig.replace('png', 'npz'), **exported_data)

def make_exp_report(
        path_log,
        path_figdir='figs',
        dpi=80):
    '''
    one exp, all values
    '''

    os.makedirs(path_figdir, exist_ok=True)

    # load log
    vals_map = load_log(path_log)

    # check all keys
    # print(' [i] (report) found keys:', ', '.join(list(vals_map.keys())))

    # plot train/valid loss
    path_figure_tr_val = os.path.join(path_figdir, 'training.png')
    xs_tr, ys_tr, ts_tr = unpack_values(vals_map['train loss'])
    is_valid = 'valid loss' in vals_map.keys()

    
    if is_valid:
        xs_va, ys_va, ts_va = unpack_values(vals_map['valid loss'])
        min_y = np.min(ys_va)
        min_x = xs_va[np.argmin(ys_va)]
        min_t = ts_va[np.argmin(ys_va)]

    plt.figure(dpi=dpi)
    plt.title('training process')
    plt.plot(xs_tr, ys_tr, label='train', alpha=0.5)
    if is_valid:
        plt.plot(xs_va, ys_va, label='valid')
        plt.plot(min_x, min_y, 'ro')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path_figure_tr_val)
    plt.close() 

    # other values
    for key in vals_map.keys():
        if key in ['valid loss', 'train loss']:
            continue
        path_fig = os.path.join(path_figdir, key + '.png')
        xs, ys, ts = unpack_values(vals_map[key])

        plt.figure(dpi=dpi)
        plt.title(key)
        plt.plot(xs, ys)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(path_fig)
        plt.close()
