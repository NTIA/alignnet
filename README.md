# Dataset Alignment
This code corresponds to the paper "AlignNet: Learning dataset score alignment functions to enable better training of speech quality estimators" Jaden Pieper, Steve Voran [Link to Paper here](XXX).

AlignNet improves the training of no-reference (NR) speech quality estimators with multiple, independent datasets. AlignNet uses an AudioNet to generate intermediate score estimates before using the Aligner to map intermediate estimates to the appropriate score range.
AlignNet is intentionally designed to be independent of the choice of AudioNet.

This repository contains implementations of two different AudioNet choices: [MOSNet](https://arxiv.org/abs/1904.08352) and a novel multi-scale convolution approach. 

MOSNet demonstrates a network that takes the STFT of an audio signal as its input and the multi-scale convolution network is provided primarily as an example of a network that takes raw audio as an input.

# Installation
## Dependencies
There are two included environment files. `environment.yml` has the dependencies required to train with alignnet, but does not impose version requirements. It is thus susceptible to issues in the future if packages deprecate methods or have major backwards compatibility breaks. On the otherhand `environment-paper.yml` contains the exact versions of the packages that were used for all the results reported in our paper. 

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
When training with multiple datasets some work must first be done to format them in a consistent manner so they can all be loaded in the same way.
For each dataset one must first make a csv that has subjective score in column called `MOS` and path to audio file in column called `audio_path`.

If your `audio_net` model requires transformed data you can transform it prior to training with `pretransform_data.py` (see `python pretransform_data.py --help` for more information) and store paths to those transformed representation files in a column called `transform_path`. For example MOSNet uses the STFT of audio as an input. For more efficient training, pretransforming the audio into STFT representations, saving them, and including a column called `stft_path` in the csv is recommended.
To use the appropriate column during training the `data.pathcol` must be set to right value.
For examples see [MOSNet](alignnet/config/models/pretrain-MOSNet.yaml) or [MultiScaleConvolution](alignnet/config/models/pretrain-msc.yaml).


For each dataset split data with
```
python split_labeled_data.py /path/to/data/file.csv --output-dir /datasetX/splits/path
```
This generates `train.csv`, `valid.csv`, and `test.csv` in `/datasetX/splits/path`.
Additional options for splitting can be seen via `python split_labeled_data.py --help`, including creating multiple independent splits and changing the amount of data placed into each split.

# Training with AlignNet

### Pretraining MOSNet on a dataset
In order to pretrain on a dataset you run
```
python /path/to/alignnet/train.py \
data.data_dirs=[datasetX/splits/path] \
--config-dir /path/to/alignnet/alignnet/config/models/ --config-name pretrain-MOSNet.yaml
```
Where `datasetX/splits/path` contains `train.csv`, `valid.csv`, and `test.csv` for that dataset.

### Training MOSNet with AlignNet
```
python /path/to/alignnet/train.py \
data.data_dirs=[/dataset1/splits/path,/dataset2/splits/path] \
--config-dir /path/to/alignnet/alignnet/config/models/ --config-name alignnet-MOSNet.yaml
```

### Training MOSNet with AlignNet and MDF
```
python /path/to/alignnet/train.py \
data.data_dirs=[dataset1/splits/path,dataset2/splits/path] \
finetune.restore_file=/path/to/alignnet/pretrained/model \
--config-dir /path/to/alignnet/alignnet/config/models/ --config-name alignnet-MOSNet.yaml
```

### Training MOSNet in conventional way
Multiple datasets, no alignment.
```
python /path/to/alignnet/train.py \
project.task=Conventional-MOSNet \
data.data_dirs=[dataset1/splits/path,dataset2/splits/path] \
--config-dir /path/to/alignnet/alignnet/config/models/ --config-name pretrain-MOSNet.yaml
```

## Examples
## Training MOSNet with AlignNet and MDF with pretraining on Tencent
```
python /path/to/alignnet/train.py \
data.data_dirs=[dataset1/splits/path,dataset2/splits/path] \
finetune.restore_file=/path/to/alignnet/trained_models/pretrained-MOSNet-tencent \
--config-dir /path/to/alignnet/alignnet/config/models/ --config-name alignnet-MOSNet.yaml
```

## MultiScaleConvolution example
Training NR speech estimators with AlignNet is intentionally designed to be agnostic to the choice of AudioNet.
To demonstrate this we include code for a rudimentary network that takes raw audio in as an input and trains separate convolutional networks on multiple time scales that are then aggregated into a single network component.
This network is defined as `alignnet.MultiScaleConvolution` and can be trained via:
```
python /path/to/alignnet/train.py \
data.data_dirs=[dataset1/splits/path,dataset2/splits/path] \
--config-dir /path/to/alignnet/alignnet/config/models/ --config-name alignnet-msc.yaml
```

# Using AlignNet models at inference

# Gathering datasets used in 2024 Conference Paper
All of the data used in the paper can be found through the following links and references.

* [Blizzard 2021](https://www.cstr.ed.ac.uk/projects/blizzard/data.html)
  *  Z.-H. Ling, X. Zhou, and S. King, "The Blizzard challenge 2021," in Proc. Blizzard Challenge Workshop, 2021.
* [Blizzard 2008](https://www.cstr.ed.ac.uk/projects/blizzard/data.html)
  * V. Karaiskos, S. King, R. A. J. Clark, and C. Mayo, "The Blizzard challenge 2008," in Proc. Blizzard Challenge Workshop, 2008.
* [FFTnet](https://gfx.cs.princeton.edu/pubs/Jin_2018_FAR/clips/)
  *  Z. Jin, A. Finkelstein, G. J. Mysore, and J. Lu, "FFTNet: a real-time speaker-dependent neural vocoder," in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing, 2018.
* [NOIZEUS](https://ecs.utdallas.edu/loizou/speech/noizeus/)
  * Y. Hu and P. Loizou, "Subjective comparison of speech enhancement algorithms," in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing, 2006.
* [VoiceMOS Challenge 2022](https://codalab.lisn.upsaclay.fr/competitions/695)
  * W. C. Huang, E. Cooper, Y. Tsao, H.-M. Wang, T. Toda, and J. Yamagishi, "The VoiceMOS Challenge 2022," in Proc. Interspeech 2022, 2022, pp. 4536–4540.
* [Tencent](https://github.com/ConferencingSpeech/ConferencingSpeech2022)
  * G. Yi, W. Xiao, Y. Xiao, B. Naderi, S. Moller, W. Wardah, G. Mittag, R. Cutler, Z. Zhang, D. S. Williamson, F. Chen, F. Yang, and S. Shang, "ConferencingSpeech 2022 Challenge: Non-intrusive objective speech quality assessment challenge for online conferencing applications," in Proc. Interspeech, 2022, pp. 3308–3312.
* [NISQA](https://github.com/gabrielmittag/NISQA/wiki/NISQA-Corpus)
  * G. Mittag, B. Naderi, A. Chehadi, and S. M ̈oller, "NISQA: A deep CNN-self-attention model for multidimensional speech quality prediction with crowdsourced datasets,” in Proc. Interspeech, 2021, pp. 2127–2131.
* [Voice Conversion Challenge 2018](https://datashare.ed.ac.uk/handle/10283/3257)
  * J. Lorenzo-Trueba, J. Yamagishi, T. Toda, D. Saito, F. Villavicencio, T. Kinnunen, and Z. Ling, “The voice conversion challenge 2018: Promoting development of parallel and nonparallel methods,” in Proc. Speaker Odyssey, 2018.
* [Indiana U. MOS](https://github.com/ConferencingSpeech/ConferencingSpeech2022)
  * X. Dong and D. S. Williamson, "A pyramid recurrent network for predicting crowdsourced speech-quality ratings of real-world signals," in Proc. Interspeech, 2020.
* [PSTN](https://github.com/ConferencingSpeech/ConferencingSpeech2022)
  * G. Mittag, R. Cutler, Y. Hosseinkashi, M. Revow, S. Srinivasan, N. Chande, and R. Aichner, “DNN no-reference PSTN speech quality prediction,” in Proc. Interspeech, 2020.