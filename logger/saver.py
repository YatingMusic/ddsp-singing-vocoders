'''
author: wayn391@mastertones
'''

from genericpath import exists
import os
import json
import time
import yaml
import logging
import datetime
import collections
import numpy as np
import matplotlib.pyplot as plt

import torch

from . import utils
from . import report


class Saver(object):
    def __init__(
            self, 
            args,
            initial_global_step=-1):

        self.expdir = args.env.expdir
        exists_ok = True if self.expdir == 'test' else False

        # cold start
        self.global_step = initial_global_step
        self.init_time = time.time()
        self.last_time = time.time()

        # makedirs
        os.makedirs(self.expdir, exist_ok=exists_ok)       

        # path
        self.path_log_value = os.path.join(self.expdir, 'log_value.txt')
        self.path_log_info = os.path.join(self.expdir, 'log_info.txt')

        # ckpt
        self.path_ckptdir = os.path.join(self.expdir, 'ckpts')
        os.makedirs(self.path_ckptdir, exist_ok=exists_ok)       

        # figs
        self.path_figdir = os.path.join(self.expdir, 'figs')

        # save config
        path_config = os.path.join(self.expdir, 'config.yaml')
        with open(path_config, "w") as out_config:
            yaml.dump(dict(args), out_config)


    def log_info(self, msg):
        '''log method'''
        if isinstance(msg, dict):
            msg_list = []
            for k, v in msg.items():
                tmp_str = ''
                if isinstance(v, int):
                    tmp_str = '{}: {:,}'.format(k, v)
                else:
                    tmp_str = '{}: {}'.format(k, v)

                msg_list.append(tmp_str)
            msg_str = '\n'.join(msg_list)
        else:
            msg_str = msg
        
        # dsplay
        print(msg_str)

        # save
        with open(self.path_log_info, 'a') as fp:
            fp.write(msg_str+'\n')

    def log_value(self, loss_dict):
        '''log method'''
        cur_time = time.time() - self.init_time
        step = self.global_step

        with open(self.path_log_value, 'a') as fp:
            for key, val in loss_dict.items():
                msg_str = '{:s} | {:.10f} | {:d} | {}\n'.format(
                    key, 
                    val, 
                    step, 
                    cur_time
                )
                fp.write(msg_str)
    
    def get_interval_time(self, update=True):
        '''time method'''
        cur_time = time.time()
        time_interval = cur_time - self.last_time
        if update:
            self.last_time = cur_time
        return time_interval

    def get_total_time(self, to_str=True):
        '''time method'''
        total_time = time.time() - self.init_time
        if to_str:
            total_time = str(datetime.timedelta(
                seconds=total_time))[:-5]
        return total_time

    def save_models(
            self, 
            model_dict, 
            postfix='', 
            to_json=False):
        '''save method'''
        for name, model in model_dict.items():
            self.save_model(
                model, 
                name,
                postfix=postfix,
                to_json=to_json)

    def save_model(
            self, 
            model, 
            name='model',
            postfix='',
            to_json=False):
        '''save method'''
        # path
        if postfix:
            postfix = '_' + postfix
        path_pt = os.path.join(
            self.path_ckptdir , name+postfix+'.pt')
        path_params = os.path.join(
            self.path_ckptdir, name+postfix+'_params.pt')
       
        # check
        print(' [*] model saved: {}'.format(path_pt))
        print(' [*] model params saved: {}'.format(path_params))

        # save
        torch.save(model, path_pt)
        torch.save(model.state_dict(), path_params)

        # to json
        if to_json:
            path_json = os.path.join(
                self.path_ckptdir , name+'.json')
            utils.to_json(path_params, path_json)

    def make_report(self):
        report.make_exp_report(
            self.path_log_value,
            path_figdir=self.path_figdir)

    def global_step_increment(self):
        self.global_step += 1


