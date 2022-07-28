# Dataset
This document describe how to prepare the dataset.

## A. Steps
1. Download audio files of MPop600 [1] Dataset from [here](), and place them under the `data` folder.
2. Run `preprocess.py` to generate mel-spetrogram files. Remember to modify the configuration.
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

## B. Information
**File Srtucture**
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

**Specification of Source Audio Files**
* format: `.WAV`
* Sampling rate: 24kHz
* Bit depth: 16bit

---

[1] (APSIPA ASC'20) [MPop600: A Mandarin Popular Song Database with Aligned Audio, Lyrics, and Musical Scores for Singing Voice Synthesis](https://ieeexplore.ieee.org/document/9306461)