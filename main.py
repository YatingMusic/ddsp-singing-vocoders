import os
import argparse
import torch

from logger import utils, report
from data_cnpop import get_data_loaders
from solver import train, test, render

from ddsp.vocoder import SawSub, SawSinSub, Sins, DWS, Full
from ddsp.loss import HybridLoss


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to config file",
    )
    parser.add_argument(
        "-s",
        "--stage",
        type=str,
        required=True,
        help="Stages. Options: training/inference",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Models. Options: SawSinSub/Sins/DWS/Full/SinsSub/SawSub",
    )
    parser.add_argument(
        "-k",
        "--model_ckpt",
        type=str,
        required=False,
        help="path to existing model ckpt",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=False,
        help="path to synthesized audio files",
    )
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    
    # load config
    args = utils.load_config(cmd.config)
    print(' > config:', cmd.config)
    print(' >    exp:', args.env.expdir)

    # load model
    model = None
    if cmd.model == 'SawSinSub':
        model = SawSinSub(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            n_mag_harmonic=args.model.n_mag_harmonic,
            n_mag_noise=args.model.n_mag_noise,
            n_harmonics=args.model.n_harmonics)

    elif cmd.model == 'Sins':
        model = Sins(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            n_harmonics=args.model.n_harmonics,
            n_mag_noise=args.model.n_mag_noise)

    elif cmd.model == 'DWS':
        model = DWS(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            num_wavetables=args.model.num_wavetables,
            len_wavetables=args.model.len_wavetables,
            is_lpf=args.model.is_lpf)

    elif cmd.model == 'Full':
        model = Full(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            n_mag_harmonic=args.model.n_mag_harmonic,
            n_mag_noise=args.model.n_mag_noise,
            n_harmonics=args.model.n_harmonics)

    elif cmd.model == 'SawSub':
        model = SawSub(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size)

    else:
        raise ValueError(f" [x] Unknown Model: {cmd.model}")
    
    # load parameters
    if cmd.model_ckpt:
        model = utils.load_model_params(
            cmd.model_ckpt, model, args.device)

    # loss
    loss_func = HybridLoss(args.loss.n_ffts)

    # device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.env.gpu)
    model.to(args.device)
    loss_func.to(args.device)

    # datas
    loader_train, loader_valid = get_data_loaders(args, whole_audio=False)

    # stage
    if cmd.stage == 'training':
        train(args, model, loss_func, loader_train, loader_valid)
    elif cmd.stage == 'validation':
        output_dir = 'valid_gen'
        if cmd.output_dir:
            output_dir = cmd.output_dir
        test(args, model, loss_func, loader_valid, path_gendir=output_dir)
    elif cmd.stage == 'inference':
        # TBD
        pass
    else:
          raise ValueError(f" [x] Unkown Stage: {cmd.stage }")
    