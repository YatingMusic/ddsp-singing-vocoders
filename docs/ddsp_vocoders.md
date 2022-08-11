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

## B. Dicsussion and Future Work

### B.1 Glitch Artifacts in Long Utterance
We found early neural vocoders having glitch artifacts in long utterances. In speech this issue might not be perceived, however in singing voice where long notes are common this becomes critical. In DDSP singing vocoders we solve it with sinusoidal excitation signals. Recently, similar findings are mentioned in [5], which claming that this method "enhances the smoothness and continuity of harmonics". 

### B.2 Buzzing Artifacts in Unvoiced/Semi-Voicesd Consonants
The buzzing artifacts of unvoiced and semi-voiced segments are generated from the harmonic part signals, and only occurs in the vocoders based on subtractive synthesizers (i.e. in model `SawSinSub`, `SawSub` and `Full`). Similar finding is discussed in [6]. Currently, we alleviate it by applying a Voiced/Unvoiced mask (UV mask) estimated from predicted signals on the harmonic part singals. 

There are some possible directions:
* Using filters of better capacity, instead of LTV-FIR
* Applying UV mask. 

### B.3 End-to-End Training
DDSP-based vocoders are data-efficient, intepretable and lightweight, hence it has a potential to be integrated with acoustic model which makes the end-to-end training of TTS or SVS possible. We could also start to rethink the role of mel-spectrograms: it could be replaced with control parameters of synthesizers, f0, UV mask and etc.

---
## C. References
[1] (ISMIR'22)[SawSing: A DDSP-based Singing Vocoder via Subtractive Sawtooth Waveform Synthesis](./ismir_22_sawsing.pdf)  
[2] (ICLR'20) [DDSP: Differentiable Digital Signal Processing](https://openreview.net/forum?id=B1x1ma4tDr)  
[3] (ICASSP'22)[Differentiable Wavetable Synthesis](https://arxiv.org/abs/2111.10003)  
[4] (ICASSP'93) [HNS: Speech modification based on a harmonic+noise model](https://ieeexplore.ieee.org/document/319365)  
[5] (ICASSP'22) [Improving adversarial waveform generation based singing voice conversion with harmonic signals](https://arxiv.org/abs/2201.10130)  
[6] (INTERSPEECH'22) [Unified Source-Filter GAN with
Harmonic-plus-Noise Source Excitation Generation](https://arxiv.org/pdf/2205.06053.pdf)
