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
please refer to [doc](./docs/dataset.md) for more details

## C. Training

Train vocoders from scratch. 
1. Please 
2. Modify the configuration file `..config/<model_name>.yaml`
3. Run the following command:
```bash
# SawSing as an example
python main.py --config ./configs/sawsinsub.yaml \
               --stage  training \
               --model SawSinSub
```
## D. Validation
Run validation: compute loss and real-time factor (RTF).

* Modify the configuration file  `..config/<model_name>.yaml`
* Run the following command:

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
* [DDSP-based Singing Voice Vocoders]()
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


