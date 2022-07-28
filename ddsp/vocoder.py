import math
import numpy as np
import soundfile as sf

import torch
import torch.nn as nn
from torch.nn import functional as F

from .mel2control import Mel2Control
from .modules import SawtoothGenerator, HarmonicOscillator, WavetableSynthesizer, WaveGeneratorOscillator
from .core import scale_function, unit_to_hz2, frequency_filter, upsample


class Full(nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_mag_harmonic,
            n_mag_noise,
            n_harmonics,
            n_mels=80):
        super().__init__()

        print(' [Model] Sinusoids Synthesiser, gt fo')

        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        
        # Mel2Control
        split_map = {
            'f0': 1, 
            'A': 1,
            'amplitudes': n_harmonics,
            'harmonic_magnitude': n_mag_harmonic,
            'noise_magnitude': n_mag_noise
        }
        self.mel2ctrl = Mel2Control(n_mels, split_map)

        # Harmonic Synthsizer
        self.harmonic_synthsizer = HarmonicOscillator(sampling_rate)

    def forward(self, mel, initial_phase=None):
        '''
            mel: B x n_frames x n_mels
        '''

        ctrls = self.mel2ctrl(mel)

        # unpack
        f0_unit = ctrls['f0']# units
        f0_unit = torch.sigmoid(f0_unit)
        f0 = unit_to_hz2(f0_unit, hz_min = 80.0, hz_max=1000.0)
        f0[f0<80] = 0

        pitch = f0
        
        A           = scale_function(ctrls['A'])
        amplitudes  = scale_function(ctrls['amplitudes'])
        src_param = scale_function(ctrls['harmonic_magnitude'])
        noise_param = scale_function(ctrls['noise_magnitude'])

        amplitudes /= amplitudes.sum(-1, keepdim=True) # to distribution
        amplitudes *= A

        # exciter signal
        B, n_frames, _ = pitch.shape
        
        # upsample
        pitch = upsample(pitch, self.block_size)
        amplitudes = upsample(amplitudes, self.block_size)

        # harmonic
        harmonic, final_phase = self.harmonic_synthsizer(
            pitch, amplitudes, initial_phase)
        harmonic = frequency_filter(
                        harmonic,
                        src_param)
            
        # noise part
        noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1
        noise = frequency_filter(
                        noise,
                        noise_param)
        signal = harmonic + noise

        return signal, f0, final_phase, (harmonic, noise)


class SawSinSub(nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_mag_harmonic,
            n_mag_noise,
            n_harmonics,
            n_mels=80):
        super().__init__()

        print(' [Model] Sawtooth (with sinusoids) Subtractive Synthesiser')

        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        
        # Mel2Control
        split_map = {
            'f0': 1, 
            'harmonic_magnitude': n_mag_harmonic,
            'noise_magnitude': n_mag_noise
        }
        self.mel2ctrl = Mel2Control(n_mels, split_map)

        # Harmonic Synthsizer
        self.harmonic_amplitudes = nn.Parameter(
            1. / torch.arange(1, n_harmonics + 1).float(), requires_grad=False)
        self.ratio = nn.Parameter(torch.tensor([0.4]).float(), requires_grad=False)

        self.harmonic_synthsizer = WaveGeneratorOscillator(
            sampling_rate,
            amplitudes=self.harmonic_amplitudes,
            ratio=self.ratio)

    def forward(self, mel, initial_phase=None):
        '''
            mel: B x n_frames x n_mels
        '''

        ctrls = self.mel2ctrl(mel)

        # unpack
        f0_unit = ctrls['f0']# units
        f0_unit = torch.sigmoid(f0_unit)
        f0 = unit_to_hz2(f0_unit, hz_min=80.0, hz_max=1000.0)
        f0[f0<80] = 0.

        pitch = f0
        
        src_param = scale_function(ctrls['harmonic_magnitude'])
        noise_param = scale_function(ctrls['noise_magnitude'])

        # exciter signal
        B, n_frames, _ = pitch.shape

        # upsample
        pitch = upsample(pitch, self.block_size)

        # harmonic
        harmonic, final_phase = self.harmonic_synthsizer(pitch, initial_phase)
        harmonic = frequency_filter(
                        harmonic,
                        src_param)

        # noise part
        noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1
        noise = frequency_filter(
                        noise,
                        noise_param)
        signal = harmonic + noise

        return signal, f0, final_phase, (harmonic, noise), # (src_param, noise_param)


class Sins(nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_harmonics,
            n_mag_noise,
            n_mels=80):
        super().__init__()

        print(' [Model] Sinusoids Synthesiser')

        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        
        # Mel2Control
        split_map = {
            'f0': 1, 
            'A': 1,
            'amplitudes': n_harmonics,
            'noise_magnitude': n_mag_noise,
        }
        self.mel2ctrl = Mel2Control(n_mels, split_map)

        # Harmonic Synthsizer
        self.harmonic_synthsizer = HarmonicOscillator(sampling_rate)

    def forward(self, mel, initial_phase=None):
        '''
            mel: B x n_frames x n_mels
        '''

        ctrls = self.mel2ctrl(mel)

        # unpack
        f0_unit = ctrls['f0']# units
        f0_unit = torch.sigmoid(f0_unit)
        f0 = unit_to_hz2(f0_unit, hz_min = 80.0, hz_max=1000.0)
        f0[f0<80] = 0

        pitch = f0
        
        A           = scale_function(ctrls['A'])
        amplitudes  = scale_function(ctrls['amplitudes'])
        noise_param = scale_function(ctrls['noise_magnitude'])

        amplitudes /= amplitudes.sum(-1, keepdim=True) # to distribution
        amplitudes *= A

        # exciter signal
        B, n_frames, _ = pitch.shape
        
        # upsample
        pitch = upsample(pitch, self.block_size)
        amplitudes = upsample(amplitudes, self.block_size)

        # harmonic
        harmonic, final_phase = self.harmonic_synthsizer(
            pitch, amplitudes, initial_phase)
            
        # noise part
        noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1
        noise = frequency_filter(
                        noise,
                        noise_param)
        signal = harmonic + noise

        return signal, f0, final_phase, (harmonic, noise) #, (noise_param, noise_param)


class DWS(nn.Module):
    def __init__(
            self,
            sampling_rate,
            block_size,            
            num_wavetables,
            len_wavetables,
            is_lpf=False):
        super().__init__()

        print(' [Model] Wavetables Synthesiser, is_lpf:', is_lpf)
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        # Mel2Control
        split_map = {
            'f0': 1, 
            'A': 1,
            'amplitudes': num_wavetables,
            'noise_magnitude': 80
        }
        self.mel2ctrl = Mel2Control(80, split_map)

        # Harmonic Synthsizer
        self.wavetables = nn.Parameter(torch.randn(num_wavetables, len_wavetables))
        self.harmonic_synthsizer = WavetableSynthesizer(
            sampling_rate, self.wavetables, block_size, is_lpf=is_lpf)

    def forward(self, mel, initial_phase=None):
        '''
            mel: B x n_frames x n_mels
        '''

        ctrls = self.mel2ctrl(mel)

        # unpack
        f0_unit = ctrls['f0']# units
        f0_unit = torch.sigmoid(f0_unit)
        f0 = unit_to_hz2(f0_unit, hz_min = 80.0, hz_max=1000.0)
        f0[f0<80] = 0

        pitch = f0
        
        A           = scale_function(ctrls['A'])
        amplitudes  = scale_function(ctrls['amplitudes'])
        noise_param = scale_function(ctrls['noise_magnitude'])

        amplitudes /= amplitudes.sum(-1, keepdim=True) # to distribution
        amplitudes *= A

        # exciter signal
        B, n_frames, _ = pitch.shape

        # harmonic
        harmonic, final_phase = self.harmonic_synthsizer(
            pitch, amplitudes, initial_phase)

        # noise part
        noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1
        noise = frequency_filter(
                        noise,
                        noise_param)
        signal = harmonic + noise

        return signal, f0, final_phase, (harmonic, noise)



class SawSub(nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,):
        super().__init__()

        print(' [Model] Sawtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        # Mel2Control
        split_map = {
            'f0': 1, 
            'harmonic_magnitude': 512, # 1024 for 48k, 512 for 24k 
            'noise_magnitude': 80
        }
        self.mel2ctrl = Mel2Control(80, split_map)

        # Harmonic Synthsizer
        self.harmonic_synthsizer = SawtoothGenerator(sampling_rate)

    def forward(self, mel, initial_phase=None):
        '''
            mel: B x n_frames x n_mels
        '''

        ctrls = self.mel2ctrl(mel)

        # unpack
        f0_unit = ctrls['f0'] # units
        f0_unit = torch.sigmoid(f0_unit)
        f0 = unit_to_hz2(f0_unit, hz_min = 80.0, hz_max=1000.0)
        f0[f0<80] = 0

        pitch = f0
        
        src_param = scale_function(ctrls['harmonic_magnitude'])
        noise_param = scale_function(ctrls['noise_magnitude'])

        # exciter signal
        B, n_frames, _ = pitch.shape

        # upsample
        pitch = upsample(pitch, self.block_size)

        # harmonic
        harmonic, final_phase = self.harmonic_synthsizer(pitch, initial_phase)
        harmonic= frequency_filter(
                        harmonic,
                        src_param)

        # noise part
        noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1
        noise = frequency_filter(
                        noise,
                        noise_param)
        signal = harmonic + noise

        return signal, f0, final_phase, (harmonic, noise)
