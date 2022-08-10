# DDSP Singing Vocoders
Authors: [Da-Yi Wu](https://github.com/ericwudayi)\*, [Wen-Yi Hsiao](https://github.com/wayne391)\*, [Fu-Rong Yang](https://github.com/furongyang)\*, [Oscar Friedman](https://github.com/OscarFree), Warren Jackson, Scott Bruzenak, Yi-Wen Liu, [Yi-Hsuan Yang](https://github.com/affige)
 
 **equal contribution*
 
 
[**Paper**](./docs/ismir_22_sawsing.pdf) | [**Demo**](https://ddspvocoder.github.io/ismir-demo/) 


Official PyTorch implementation of ISMIR2022 paper "DDSP-based Singing Vocoders: A New Subtractive-based Synthesizer and A Comprehensive Evaluation".

In this repository:
* We propose a novel singing vocoders based on subtractive synthesizer: **SawSing**
* We present a collection of different ddsp singing vocoders
* We demonstrate that ddsp singing vocoders have relatively small model size but can generate satisfying results with limited resources (1 GPU, 3-hour training data). We also report the result of an even more stringent case training the vocoders with only 3-min training recordings for only 3-hour training time.

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
              --input_dir  ./path/to/mel
              --output_dir ./test_gen
```

## F. Post-Processing
In Sawsing, we found there are buzzing artifacta in the harmonic part singals, so we develop a post-processing codes to remove it. The method is simple yet effective --- applying a voiced/unvoiced mask. For more details, please refer to [here](./postprocessing/).


## G. More Information
* Checkpoints
  * **Sins (DDSP-Add)**:  [`./exp/f1-full/sins/ckpts/`](./exp/f1-full/sins/ckpts/)
  * **SawSinSub (Sawsing)**:  [`./exp/f1-full/sawsinsub-256/ckpts/`](./exp/f1-full/sawsinsub-256/ckpts/)
  * The full experimental records, reports and checkpoints can be found under the [`exp`](./exp/) folder.
* Documentation
  * [DDSP Vocoders](./docs/ddsp_vocoders.md)
  * [Synthesizer Design](./docs/synth_demo.ipynb)

## H. Citation
```
@article{sawsing,
  title={SawSing: A DDSP-based Singing Vocoder via Subtractive Sawtooth Waveform Synthesis},
  author={Da-Yi Wu, Wen-Yi Hsiao, Fu-Rong Yang, Oscar Friedman, Warren Jackson, Scott Bruzenak, Yi-Wen Liu, Yi-Hsuan Yang},
  journal = {Proc. International Society for Music Information Retrieval},
  year    = {2022},
}
```


