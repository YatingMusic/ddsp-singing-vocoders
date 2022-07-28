# DDSP Singing Voice Vocoders
Authors: Da-Yi Wu, Wen-Yi Hsiao, Fu-Rong Yang, Oscar Friedman, Warren Jackson, Scott Bruzenak, Yi-Wen Liu, Yi-Hsuan Yang

[**Paper**]() | [**Audio Demo**]() 


Official PyTorch implementation of ISMIR2022 paper "SawSing: A DDSP-based Singing Vocoder via Subtractive Sawtooth Waveform Synthesis".

A collection of ddsp-based singing voice vocoders.




## A. Installation
```bash
pip install -r requirements.txt 
```
## B. Dataset
please refer to [doc](./docs/dataset.md) for more details

## C. Training

Train vocoders from scratch. 
1. Modify `..config/sawsinsub.yaml`
2. Run the following command:
```bash
python main.py --config ./configs/sawsinsub.yaml \
               --stage  training \
               --model SawSinSub
```
## D. Validation
Run validation: compute loss and real-time factor (RTF).

* modify `..config/sawsinsub.yaml`
* run the following command:

```bash
python main.py --config ./configs/sawsinsub.yaml  \
              --stage inference \
              --model SawSinSub \
              --model_ckpt ./exp/f1-full/sawsinsub-256/ckpts/vocoder_27740_70.0_params.pt \
              --output_dir ./test_gen
```
## E. Inference
Synthesize audio file from mel-spectrogram

```bash
python main.py --config ./configs/sawsinsub.yaml  \
              --stage inference \
              --model SawSinSub \
              --model_ckpt ./exp/f1-full/sawsinsub-256/ckpts/vocoder_27740_70.0_params.pt \
              --input_dir  ./
              --output_dir ./test_gen
```

## F. More Information
* About [Synthesizer Design](./docs/synthesizer_design.md)
* About [DDSP-based Singing Voice Vocoders]()


## G. Citing
```
```


