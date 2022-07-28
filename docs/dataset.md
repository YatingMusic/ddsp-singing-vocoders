# Dataset
This document describe how to prepare the dataset.

## A. Preparation

Due to the copyright issue, we cannot distrubute audio files and the mel-spectrograms of MPop600[1] dataset. However, readers can following the steps to build one from scratch. 

### 1. File Strutcure
Organized the dataset as the following (or similar) structure.
```
data          
├─ f1                 
├─ m1     
│  ├─ test
│  ├─ train-3min
│  ├─ train-full
│  ├─ train-val
│  │  ├─  audio
│  │  │  ├─  xxx.wav
│  │  ├─  mel
│  │  │  ├─  xxx.npy
```
In this example, we have one female (`f1`) and a male (`m1`) singer. 
For each singer, we have 2 training sets for different scenatios: `train-full` (6 hours), and `train-3min` (3 minutes only). The two scenatios use the same validation (`train-val`) and testing set (`test`).

The specification of the audio files:
* format: `.WAV`
* Sampling rate: 24kHz
* Bit depth: 16bit

### 2. Preprocessing
Run `preprocess.py` to generate mel-spetrogram files, which is stored in `.npy` format. Remember to modify the configuration in `preprocess.py` at first.
```python
'''in preprocess.py'''

# ...
# ==================================================== #
# configuration
# ==================================================== #
path_rootdir = './data'
device = 'cuda'

sampling_rate  = 24000
hop_length     = 240
win_length     = 1024
n_mel_channels = 80

src_ext = 'wav'
dst_ext = 'npy'
# ...
```

### 3. Configuration
Before training, remember to change the dataset path in the configuration file, which can be founder under `configs` folder and stored in `.yaml` format.

---
## B. Reference
[1] (APSIPA ASC'20) [MPop600: A Mandarin Popular Song Database with Aligned Audio, Lyrics, and Musical Scores for Singing Voice Synthesis](https://ieeexplore.ieee.org/document/9306461)