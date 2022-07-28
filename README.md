# DDSP Singing Voice Vocoders

A collection of ddsp-based singing voice vocoders.

## SawSing
* Installation
```bash
pip install -r requirements.txt 
```
* Dataset
    * please refer to [doc](./docs/dataset.md) for more details
* Training
    * traing vocoders from scratch
    * modify `..config/sawsinsub.yaml`
    * run the following command:
```bash
python main.py --config ./configs/sawsinsub.yaml \
               --stage  training \
               --model SawSinSub
```
* Evaluation
    * run validatio
```
ython main.py --config ./configs/sawsinsub.yaml  \
              --stage inference \
              --model SawSinSub \
              --model_ckpt ./exp/f1-full/sawsinsub-256/ckpts/vocoder_27740_70.0_params.pt \
              --output_dir ./test_gen

```
* Inference

## More Information
* About [Synthesizer Design](./docs/synthesizer_design.md)
* About [DDSP-based Singing Voice Vocoders]()


---

Authors: Da-Yi Wu, Wen-Yi Hsiao, Fu-Rong Yang, Oscar Friedman, Warren Jackson, Scott Bruzenak, Yi-Wen Liu, Yi-Hsuan Yang

[Paper]() | [Audio Demo]()

Official PyTorch implementation of ISMIR2022 paper "SawSing: A DDSP-based Singing Vocoder via Subtractive Sawtooth Waveform Synthesis".
