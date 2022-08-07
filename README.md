# DDSP Singing Vocoders
Authors: Da-Yi Wu\*, [Wen-Yi Hsiao](https://github.com/wayne391)\*, [Fu-Rong Yang](https://github.com/furongyang)\*, [Oscar Friedman](https://github.com/OscarFree), Warren Jackson, Scott Bruzenak, Yi-Wen Liu, [Yi-Hsuan Yang](https://github.com/affige)
 
 **equal contribution*
 
 
[**Paper**](./docs/ismir_22_sawsing.pdf) | [**Demo**](https://ddspvocoder.github.io/ismir-demo/) 


Official PyTorch implementation of ISMIR2022 paper "SawSing: A DDSP-based Singing Vocoder via Subtractive Sawtooth Waveform Synthesis".

In this repository, we present a collection of ddsp-based vocoders for singing voice. Our experiments shows that DDSP-based vocoders can generate satisfying results with limited resources (1 GPU, 3-hour training data).

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
3. Change `--model` argument to try different vocoders. Currently, we have 5 models: `SawSinSub` (Sawsing), `Sins` (DDSP-Add), ` DWS` (DWTS), `Full`, ` SawSub`. For more details, please refer to our documentation - [DDSP Vocoders](./docs/ddsp_vocoders.md).

About our training resources:
<GPU>

## D. Validation
Run validation: compute loss and real-time factor (RTF).

1. Modify the configuration file  `..config/<model_name>.yaml`
2. Run the following command:

```bash
# SawSing as an example
python main.py --config ./configs/sawsinsub.yaml  \
              --stage validation \
              --model SawSinSub \
              --model_ckpt ./exp/f1-full/sawsinsub-256/ckpts/vocoder_27740_70.0_params.pt \
              --output_dir ./test_gen
```
## E. Inference
Synthesize audio file from existed mel-spectrograms. The code and specfication for extracting mel-spectrograms can be found in [`preprocess.py`](./preprocess.py). 

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
* Checkpoints
  * [SawSing](./exp/f1-full/sawsinsub-256/ckpts/)
  * The full experiment records, reports and checkpoints can be found under the [`exp`](./exp/) folder.
* Documentation
  * [DDSP Vocoders](./docs/ddsp_vocoders.md)
  * [Synthesizer Design](./docs/synth_demo.ipynb)

## G. Citation
```
@article{diffsynth,
  title={SawSing: A DDSP-based Singing Vocoder via Subtractive Sawtooth Waveform Synthesis},
  author={Da-Yi Wu, Wen-Yi Hsiao, Fu-Rong Yang, Oscar Friedman, Warren Jackson, Scott Bruzenak, Yi-Wen Liu, Yi-Hsuan Yang},
  booktitle = {Proc. International Society for Music Information Retrieval},
  year      = {2022},
}
```


