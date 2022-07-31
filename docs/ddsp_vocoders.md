# DDSP Vocoders

## A. Meet Our Vocoders
All the vocoders we presented are **'harmonic plus noise'** models[4]. According to their ways to modeling harmonic parts of singing voices, we have the following models:

|  Model Name  | Synthesizer          |   Note              |
|:------------:|:--------------------:|:-------------------:|
| SawSinSub    | Filtered Sawtooth Synthesizer </br>(Appproximated by Harmonic Synthesizer)| proposed SawSing[1]|
| Sins         | Harmonic Synthesizer            | [2] |
| DWS          | Wavetable Synthesizer           | [3]               |
| Full         | Filtered Harmonic Synthesizer   | modified from [2] |
| SawSub       | Filtered Sawtooth Synthesizer   | modified from [1]|

In our paper, we only compare and report 3 vocoders: Sins (DDSP-Add), DWS (DWTS), SawSinSub (SawSing). 

To try different vocoders, please modify the `--model` argument when entering commands. Note that depending on the vocoders, their configuration are slightly different.

```
python main.py --config <path-to-config> \
               --stage  training \
               --model <model-name>
```

There are two vocoders based on sawtooth synthesizer: `SawSinSub` and `SawSub`. `SawSub` generate source signals with sawtooth waveform generator without anti-aliasing. `SawSinSub` can be seen as the anti-aliased version of the previous one. It uses "Harmonic Synthesizer" with predefined coefficients of harmonics to approximate sawtooth waveform. If you would like to know more about impelementation details of syntehsizers, please refet to [synthesizer_demo](./synth_demo.ipynb). 


---
## B. References
[1] (ISMIR'22)[SawSing: A DDSP-based Singing Vocoder via Subtractive Sawtooth Waveform Synthesis](./ismir_22_sawsing.pdf)  
[2] (ICLR'20) [DDSP: Differentiable Digital Signal Processing](https://openreview.net/forum?id=B1x1ma4tDr)  
[3] (ICASSP'22)[Differentiable Wavetable Synthesis](https://arxiv.org/abs/2111.10003)  
[4] (ICASSP'93) [HNS: Speech modification based on a harmonic+noise model](https://ieeexplore.ieee.org/document/319365)  