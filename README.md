# Dataset Alignment
[![DOI](https://zenodo.org/badge/800195387.svg)](https://zenodo.org/doi/10.5281/zenodo.12734153)

This code corresponds to the paper "AlignNet: Learning dataset score alignment functions to enable better training of speech quality estimators," by Jaden Pieper, Stephen D. Voran, to appear in Proc. Interspeech 2024 and with [preprint available here](https://arxiv.org/abs/2406.10205).

When training a no-reference (NR) speech quality estimator, multiple datasets provide more information and can thus lead to better training. But they often are inconsistent in the sense that they use different subjective testing scales, or the exact same scale is used differently by test subjects due to the corpus effect.
AlignNet improves the training of NR speech quality estimators with multiple, independent datasets. AlignNet uses an AudioNet to generate intermediate score estimates before using the Aligner to map intermediate estimates to the appropriate score range.
AlignNet is intentionally designed to be independent of the choice of AudioNet.

This repository contains implementations of two different AudioNet choices: [MOSNet](https://arxiv.org/abs/1904.08352) and a simple example of a novel multi-scale convolution approach. 

MOSNet demonstrates a network that takes the STFT of an audio signal as its input, and the multi-scale convolution network is provided primarily as an example of a network that takes raw audio as an input.

# Installation
## Dependencies
There are two included environment files. `environment.yml` has the dependencies required to train with alignnet but does not impose version requirements. It is thus susceptible to issues in the future if packages deprecate methods or have major backwards compatibility breaks. On the other hand, `environment-paper.yml` contains the exact versions of the packages that were used for all the results reported in our paper. 

Create and activate the `alignnet` environment.
```
conda env create -f environment.yml
conda activate alignnet
```

## Installing alignnet package
```
pip install .
```

# Preparing data for training
When training with multiple datasets, some work must first be done to format them in a consistent manner so they can all be loaded in the same way.
For each dataset, one must first make a csv that has subjective score in column called `MOS` and path to audio file in column called `audio_path`.

If your `audio_net` model requires transformed data, you can transform it prior to training with `pretransform_data.py` (see `python pretransform_data.py --help` for more information) and store paths to those transformed representation files in a column called `transform_path`. For example, MOSNet uses the STFT of audio as an input. For more efficient training, pretransforming the audio into STFT representations, saving them, and including a column called `stft_path` in the csv is recommended.
More generally, the column name must match the value of `data.pathcol`.
For examples, see [MOSNet](alignnet/config/models/pretrain-MOSNet.yaml) or [MultiScaleConvolution](alignnet/config/models/pretrain-msc.yaml).


For each dataset, split the data into training, validation, and testing portions with
```
python split_labeled_data.py /path/to/data/file.csv --output-dir /datasetX/splits/path
```
This generates `train.csv`, `valid.csv`, and `test.csv` in `/datasetX/splits/path`.
Additional options for splitting can be seen via `python split_labeled_data.py --help`, including creating multiple independent splits and changing the amount of data placed into each split.

# Training with AlignNet
Setting up training runs is configured via [Hydra](https://hydra.cc/docs/intro/).
Basic examples of configuration files can be found in [model/config](alignnet/config/models).

Some basic training help can be found with 

```
python train.py --help
```

To see an example config file and all the overrideable parameters for training MOSNet with AlignNet, run
```
python train.py --config-dir alignnet/config/models --config-name=alignnet-MOSNet --cfg job
```
Here the `--cfg job` shows the configuration for this job without running the code.

If you are not training with a [clearML](https://clear.ml/) server, be sure to set `logging=none`.
To change the number of workers used for data loading, override the `data.num_workers` parameter, which defaults to 6.

As an example, and to confirm you have appropriately overridden these parameters, you could run 
```
python train.py logging=none data.num_workers=4 --config-dir alignnet/config/models --config-name=alignnet-MOSNet --cfg job
```

### Pretraining MOSNet on a dataset
In order to pretrain on a dataset you run
```
python path/to/alignnet/train.py \
data.data_dirs=[/absolute/path/datasetX/splits/path] \
--config-dir path/to/alignnet/alignnet/config/models/ --config-name pretrain-MOSNet.yaml
```
Where `/absolute/path/datasetX/splits/path` contains `train.csv`, `valid.csv`, and `test.csv` for that dataset.

### Training MOSNet with AlignNet
```
python path/to/alignnet/train.py \
data.data_dirs=[/absolute/path/dataset1/splits/path,/absolute/path/dataset2/splits/path] \
--config-dir path/to/alignnet/alignnet/config/models/ --config-name alignnet-MOSNet.yaml
```

### Training MOSNet with AlignNet and MDF
```
python path/to/alignnet/train.py \
data.data_dirs=[/absolute/path/dataset1/splits/path,/absolute/path/dataset2/splits/path] \
finetune.restore_file=/absolute/path/to/alignnet/pretrained/model \
--config-dir path/to/alignnet/alignnet/config/models/ --config-name alignnet-MOSNet.yaml
```

### Training MOSNet in conventional way
Multiple datasets, no alignment.
```
python path/to/alignnet/train.py \
project.task=Conventional-MOSNet \
data.data_dirs=[/absolute/path/dataset1/splits/path,/absolute/path/dataset2/splits/path] \
--config-dir path/to/alignnet/alignnet/config/models/ --config-name pretrain-MOSNet.yaml
```

## Examples
## Training MOSNet with AlignNet and MDF starting with MOSNet that has been pretrained on Tencent dataset
```
python path/to/alignnet/train.py \
data.data_dirs=[/absolute/path/dataset1/splits/path,/absolute/path/dataset2/splits/path] \
finetune.restore_file=/absolute/path/to/alignnet/trained_models/pretrained-MOSNet-tencent \
--config-dir path/to/alignnet/alignnet/config/models/ --config-name alignnet-MOSNet.yaml
```

## MultiScaleConvolution example
Training NR speech estimators with AlignNet is intentionally designed to be agnostic to the choice of AudioNet.
To demonstrate this, we include code for a rudimentary network that takes in raw audio as an input and trains separate convolutional networks on multiple time scales that are then aggregated into a single network component.
This network is defined as `alignnet.MultiScaleConvolution` and can be trained via:
```
python path/to/alignnet/train.py \
data.data_dirs=[/absolute/path/dataset1/splits/path,/absolute/path/dataset2/splits/path] \
--config-dir path/to/alignnet/alignnet/config/models/ --config-name alignnet-msc.yaml
```

# Using AlignNet models at inference
Trained AlignNet models can easily be used at inference via the CLI built into `inference.py`.
Some basic help can be seen via
```
python inference.py --help
```

In general, three overrides must be set:
* `model.path` - path to a trained model
* `data.data_files` - list containing absolute paths to csv files that list audio files to perform inference on.
* `output.file` - path to file where inference output will be stored.

After running inference, a csv will be created at `output.file` with the following columns:
* `file` - filenames where audio was loaded from
* `estimate` - estimate generated by the model
* `dataset` - index listing which file from `data.data_files` this file belongs to.
* `AlignNet dataset index` - index listing which dataset within the model the scores come from. This will be the same for every file in the csv. The default dataset will always be the reference dataset, but this can be overriden via `model.dataset_index`.

For example, to run inference using the included AlignNet model trained on the smaller datasets, one would run
```
python inference.py \
data.data_files=[/absolute/path/to/inference/data1.csv,/absolute/path/to/inference/data2.csv] \
model.path=trained_models/alignnet_mdf-MOSNet-small_data \
output.file=estimations.csv
```


# Gathering datasets used in 2024 Conference Paper
Here are links and references to help with locating the data we have used in the paper.

* [Blizzard 2021](https://www.cstr.ed.ac.uk/projects/blizzard/data.html)
  *  Z.-H. Ling, X. Zhou, and S. King, "The Blizzard challenge 2021," in Proc. Blizzard Challenge Workshop, 2021.
* [Blizzard 2008](https://www.cstr.ed.ac.uk/projects/blizzard/data.html)
  * V. Karaiskos, S. King, R. A. J. Clark, and C. Mayo, "The Blizzard challenge 2008," in Proc. Blizzard Challenge Workshop, 2008.
* [FFTNet](https://gfx.cs.princeton.edu/pubs/Jin_2018_FAR/clips/)
  *  Z. Jin, A. Finkelstein, G. J. Mysore, and J. Lu, "FFTNet: a real-time speaker-dependent neural vocoder," in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing, 2018.
* [NOIZEUS](https://ecs.utdallas.edu/loizou/speech/noizeus/)
  * Y. Hu and P. Loizou, "Subjective comparison of speech enhancement algorithms," in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing, 2006.
* [VoiceMOS Challenge 2022](https://codalab.lisn.upsaclay.fr/competitions/695)
  * W. C. Huang, E. Cooper, Y. Tsao, H.-M. Wang, T. Toda, and J. Yamagishi, "The VoiceMOS Challenge 2022," in Proc. Interspeech 2022, 2022, pp. 4536–4540.
* [Tencent](https://github.com/ConferencingSpeech/ConferencingSpeech2022)
  * G. Yi, W. Xiao, Y. Xiao, B. Naderi, S. Moller, W. Wardah, G. Mittag, R. Cutler, Z. Zhang, D. S. Williamson, F. Chen, F. Yang, and S. Shang, "ConferencingSpeech 2022 Challenge: Non-intrusive objective speech quality assessment challenge for online conferencing applications," in Proc. Interspeech, 2022, pp. 3308–3312.
* [NISQA](https://github.com/gabrielmittag/NISQA/wiki/NISQA-Corpus)
  * G. Mittag, B. Naderi, A. Chehadi, and S. Möller, "NISQA: A deep CNN-self-attention model for multidimensional speech quality prediction with crowdsourced datasets,” in Proc. Interspeech, 2021, pp. 2127–2131.
* [Voice Conversion Challenge 2018](https://datashare.ed.ac.uk/handle/10283/3257)
  * J. Lorenzo-Trueba, J. Yamagishi, T. Toda, D. Saito, F. Villavicencio, T. Kinnunen, and Z. Ling, “The voice conversion challenge 2018: Promoting development of parallel and nonparallel methods,” in Proc. Speaker Odyssey, 2018.
* [Indiana U. MOS](https://github.com/ConferencingSpeech/ConferencingSpeech2022)
  * X. Dong and D. S. Williamson, "A pyramid recurrent network for predicting crowdsourced speech-quality ratings of real-world signals," in Proc. Interspeech, 2020.
* [PSTN](https://github.com/ConferencingSpeech/ConferencingSpeech2022)
  * G. Mittag, R. Cutler, Y. Hosseinkashi, M. Revow, S. Srinivasan, N. Chande, and R. Aichner, “DNN no-reference PSTN speech quality prediction,” in Proc. Interspeech, 2020.
