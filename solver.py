import os
import sys
import time
import shutil

import torch

import numpy as np
from logger.saver import Saver
from logger import utils

import soundfile as sf


def half_learning_rate(optimizer):
    old_lr = None
    for param_group in optimizer.param_groups:
        old_lr = param_group['lr']
        new_lr = old_lr  * 0.5
        param_group['lr'] = new_lr
    return old_lr, new_lr


def render(args, model, path_mel_dir, dirname='gen'):
    print(' [*] rendering...')
    model.eval()

    files = utils.traverse_dir(
        path_mel_dir, 
        extension='npy', 
        is_sort=True, 
        is_pure=True)
    num_files = len(files)

    path_gendir = os.path.join(args.env.expdir, dirname)
    os.makedirs(path_gendir, exist_ok=True)
    rtf_all = []
    with torch.no_grad():
        for fidx in range(num_files):
            fn = files[fidx]
            print('--------')
            print('{}/{} - {}'.format(fidx, num_files, fn))

            path_mel = os.path.join(path_mel_dir, fn)
            mel = np.load(path_mel)
            mel = torch.from_numpy(mel).float().to(args.device).unsqueeze(0)
            print(' mel:', mel.shape)

            # forward
            signal, f0_pred, _, (s_h, s_n) = model(mel)

            # save
            path_pred = os.path.join(path_gendir, 'pred', fn + '.wav')
            print(' > path_pred:', path_pred)
            
            os.makedirs(os.path.dirname(path_pred), exist_ok=True)
            pred   = utils.convert_tensor_to_numpy(signal)
            print('pred:', pred.shape)
            
            sf.write(path_pred, pred, args.data.sampling_rate)


def test(args, model, loss_func, loader_test, dirname='gen'):
    print(' [*] testing...')
    model.eval()

    test_loss = 0.
    test_loss_mss = 0.
    test_loss_f0 = 0.
    
    num_batches = len(loader_test)

    path_gendir = os.path.join(args.env.expdir, dirname)
    os.makedirs(path_gendir, exist_ok=True)
    rtf_all = []
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if k != 'name':
                    data[k] = data[k].to(args.device).float()
            print('>>', data['name'][0])

            # forward
            st_time = time.time()
            signal, f0_pred, _, (s_h, s_n) = model(data['mel'])
            ed_time = time.time()

        
            # crop
            # print(signal.shape, data['audio'].shape)
            min_len = np.min([signal.shape[1], data['audio'].shape[1]])
            signal        = signal[:,:min_len]
            data['audio'] = data['audio'][:,:min_len]

            # RTF
            run_time = ed_time - st_time
            song_time = data['audio'].shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            rtf_all.append(rtf)
           
            # loss
            loss, (loss_mss, loss_f0) = loss_func(
                signal, data['audio'], f0_pred, data['f0'])

            test_loss         += loss.item()
            test_loss_mss     += loss_mss.item() 
            test_loss_f0      += loss_f0.item()

            # save
            path_pred = os.path.join(path_gendir, 'pred', fn + '.wav')
            path_anno = os.path.join(path_gendir, 'anno', fn + '.wav')

            print(' > path_pred:', path_pred)
            print(' > path_anno:', path_anno)

            os.makedirs(os.path.dirname(path_pred), exist_ok=True)
            os.makedirs(os.path.dirname(path_anno), exist_ok=True)

            pred   = utils.convert_tensor_to_numpy(signal)
            anno   = utils.convert_tensor_to_numpy(data['audio'])
            
            sf.write(path_pred, pred, args.data.sampling_rate)
            sf.write(path_anno, anno, args.data.sampling_rate)

    # report
    test_loss /= num_batches
    test_loss_mss     /= num_batches
    test_loss_f0      /= num_batches

    print(' [test_loss] test_loss:', test_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    return test_loss, test_loss_mss, test_loss_f0


def train(args, model, loss_func, loader_train, loader_test):
    # saver
    saver = Saver(args)

    # model size
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.lr)

    # run
    best_loss = np.inf
    num_batches = len(loader_train)
    model.train()
    prev_save_time = -1
    saver.log_info('======= start training =======')
    for epoch in range(args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()

            # unpack data
            for k in data.keys():
                if k != 'name':
                    data[k] = data[k].to(args.device).float()
            
            # forward
            signal, f0_pred, _, _,  = model(data['mel'])

            # loss
            loss, (loss_mss, loss_f0) = loss_func(
                signal, data['audio'], f0_pred, data['f0'])
            
            # handle nan loss
            if torch.isnan(loss):
                raise ValueError(' [x] nan loss ')
            else:
                # backpropagate
                loss.backward()
                optimizer.step()

            # log loss
            if saver.global_step % args.train.interval_log == 0:
                saver.log_info(
                    'epoch: {}/{} {:3d}/{:3d} | {} | t: {:.2f} | loss: {:.6f} | time: {} | counter: {}'.format(
                        epoch,
                        args.train.epochs,
                        batch_idx,
                        num_batches,
                        saver.expdir,
                        saver.get_interval_time(),
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                saver.log_info(
                    ' > mss loss: {:.6f}, f0: {:.6f}'.format(
                       loss_mss.item(),
                       loss_f0.item(),
                    )
                )

                y, s = signal, data['audio']
                saver.log_info(
                    "pred: max:{:.5f}, min:{:.5f}, mean:{:.5f}, rms: {:.5f}\n" \
                    "anno: max:{:.5f}, min:{:.5f}, mean:{:.5f}, rms: {:.5f}".format(
                            torch.max(y), torch.min(y), torch.mean(y), torch.mean(y** 2) ** 0.5,
                            torch.max(s), torch.min(s), torch.mean(s), torch.mean(s** 2) ** 0.5))

                saver.log_value({
                    'train loss': loss.item(), 
                    'train loss mss': loss_mss.item(),
                    'train loss f0': loss_f0.item(),
                })
            
            # validation
            # if saver.global_step % args.train.interval_val == 0:
            cur_hour = saver.get_total_time(to_str=False) // 3600
            if cur_hour != prev_save_time:
                # save latest
                saver.save_models(
                        {'vocoder': model}, postfix=f'{saver.global_step}_{cur_hour}')

                prev_save_time = cur_hour
                # run testing set
                test_loss, test_loss_mss, test_loss_f0 = test(
                    args, model, loss_func, loader_test,
                     dirname=os.path.join(
                        'runtime_gen', f'gen_{saver.global_step}_{cur_hour}'))
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.6f}. mss loss: {:.6f}, f0: {:.6f}'.format(
                        test_loss, test_loss_mss, test_loss_f0
                    )
                )

                saver.log_value({
                    'valid loss': test_loss,
                    'valid loss mss': test_loss_mss,
                    'valid loss f0': test_loss_f0,
                })
                model.train()

                # save best model
                if test_loss < best_loss:
                    saver.log_info(' [V] best model updated.')
                    saver.save_models(
                        {'vocoder': model}, postfix='best')
                    test_loss = best_loss

                saver.make_report()

                          
