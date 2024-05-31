# Dataset Alignment
Speech quality estimation that uses dataset identifiers to separately learn:
* mappings between audio and a reference dataset score space
* mappings between the reference dataset score space and score spaces for every other dataset.

## Dependencies
Create and activate the `alignnet` environment.
```
conda env create -f environment.yml
conda activate alignnet
```

## Datasets Used in 2024 Conference Paper
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

## General training commands

### Pretraining MOSNet on a dataset
In order to pretrain on a dataset you run
```
python /path/to/alignnet/train.py \
data.dirs=[/path/to/datasetX] \
--config-dir /path/to/alignnet/config/models/ --config-name pretrain-MOSNet.yaml \
```
Where `/path/to/datasetX` contains `train.csv`, `valid.csv`, and `test.csv` for that dataset.

### Training MOSNet with AlignNet
```
python /path/to/alignnet/train.py \
data.dirs=[/path/to/dataset1,/path/to/dataset2] \
--config-dir /path/to/alignnet/config/models/ --config-name alignnet-MOSNet.yaml \
```

### Training MOSNet with AlignNet and MDF
```
python /path/to/alignnet/train.py \
data.dirs=[/path/to/dataset1,/path/to/dataset2] \
checkpoint.restore_file=/path/to/pretrained/model \
--config-dir /path/to/alignnet/config/models/ --config-name alignnet-MOSNet.yaml \
```

## Examples
## Training MOSNet with AlignNet and MDF with pretraining on Tencent
```
python /path/to/alignnet/train.py \
data.dirs=[/path/to/dataset1,/path/to/dataset2] \
checkpoint.restore_file=trained_models/pretrained-MOSNet-tencent \
--config-dir /path/to/alignnet/config/models/ --config-name alignnet-MOSNet.yaml
```
