# DDSP Singing Voice Vocoders
Authors: Da-Yi Wu\*, Wen-Yi Hsiao\*, Fu-Rong Yang\*, Oscar Friedman, Warren Jackson, Scott Bruzenak, Yi-Wen Liu, Yi-Hsuan Yang

[**Paper**]() | [**Audio Demo**]() 


Official PyTorch implementation of ISMIR2022 paper "SawSing: A DDSP-based Singing Vocoder via Subtractive Sawtooth Waveform Synthesis".

In this repository:
1. We present collection of ddsp-based singing voice vocoders.
2. We demonstrate the design of synthesizers and vocoders in PyTorch.

## A. Installation
```bash
pip install -r requirements.txt 
```
## B. Dataset
Please refer to [dataset.md](./docs/dataset.md) for more details.

## C. Training

Train vocoders from scratch. 
1. Modify the configuration file `..config/<model_name>.yaml`
2. Run the following command:
```bash
# SawSing as an example
python main.py --config ./configs/sawsinsub.yaml \
               --stage  training \
               --model SawSinSub
```
3. Change `--model` argument to try different vocoders. Currently, we have 5 options: `SawSinSub` (Sawsing), `Sins` (DDSP-Add), ` DWS` (DWTS), `Full`, ` SawSub`. For more details, please refer to our documentation - [DDSP Vocoders](./docs/ddsp_vocoders.md).


## D. Validation
Run validation: compute loss and real-time factor (RTF).

1. Modify the configuration file  `..config/<model_name>.yaml`
2. Run the following command:

```bash
# SawSing as an example
python main.py --config ./configs/sawsinsub.yaml  \
              --stage inference \
              --model SawSinSub \
              --model_ckpt ./exp/f1-full/sawsinsub-256/ckpts/vocoder_27740_70.0_params.pt \
              --output_dir ./test_gen
```
## E. Inference
Synthesize audio file from existed mel-spectrograms.

```bash
# SawSing as an example
python main.py --config ./configs/sawsinsub.yaml  \
              --stage inference \
              --model SawSinSub \
              --model_ckpt ./exp/f1-full/sawsinsub-256/ckpts/vocoder_27740_70.0_params.pt \
              --input_dir  ./
              --output_dir ./test_gen
```

## F. More Information
* [DDSP Vocoders](./docs/ddsp_vocoders.md)
* [Synthesizer Design](./docs/synthesizer_design.md)

## G. Citing
```
@article{diffsynth,
  title={SawSing: A DDSP-based Singing Vocoder via Subtractive Sawtooth Waveform Synthesis},
  author={Da-Yi Wu, Wen-Yi Hsiao, Fu-Rong Yang, Oscar Friedman, Warren Jackson, Scott Bruzenak, Yi-Wen Liu, Yi-Hsuan Yang},
  booktitle = {Proc. International Society for Music Information Retrieval},
  year      = {2022},
}

```


