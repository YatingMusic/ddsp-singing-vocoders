# SawSing: A DDSP-based Singing Vocoder via Subtractive Sawtooth Waveform Synthesis

Authors: Da-Yi Wu, Wen-Yi Hsiao, Fu-Rong Yang, Oscar Friedman, Warren Jackson, Scott Bruzenak, Yi-Wen Liu, Yi-Hsuan Yang

[Paper]() | [Audio Demo]()

Official PyTorch implementation of ISMIR2022 paper "SawSing: A DDSP-based Singing Vocoder via Subtractive Sawtooth Waveform Synthesis".

*some intro*

## SawSing
* Installation
```bash
pip install -r requirements.txt 
```
* Dataset
    * read the [doc](./docs/dataset.md)
* Training
    * modify `..config/sawsinsub.yaml`
    * run the following command:
```bash
python main.py --config ./configs/sawsinsub.yaml \
               --stage  training \
               --model SawSinSub
python main.py --config ./configs/sins.yaml \
               --stage  training \
               --model Sins
```
* Testing
* Inference

## More Information
* About [Synthesizer Design](./docs/synthesizer_design.md)
* About [DDSP-based Singing Voice Vocoders]()
