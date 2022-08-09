# Post-Processing of SawSing



After finishing the paper, we found that the buzzing artifacts of unvoiced and semi-voiced segments are from the harmonic part signals, and only occurs in the ones based on subtractive synthesizers (i.e. in model `SawSinSub`, `SawSub` and `Full`). To remove them, we apply a Voiced/Unvoiced mask on the harmonic part singals. The mask is currently estimated from the ground-truth, in the future this could be integrated as a part of the model, or given by acoustic models. Note that this step is excluded from the user study in our paper.

### Usage
* Run [`debuzz.ipynb`](./debuzz.ipynb)
* In the `test-set` folder:
    * `debuzz`: post-processed de-buzzing results
    * `part`: separated harmonic and noise part signals
    * `pred`: orignal predicted signals

Special thanks to [Oscar Friedman](https://github.com/OscarFree).
