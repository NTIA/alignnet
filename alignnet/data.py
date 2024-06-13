import os
import pickle
import re
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
import torchaudio
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class AudioData(Dataset):
    def __init__(
        self,
        data_files,
        transform=None,
        transform_time="get",
        cache=False,
        target="MOS",
        target_transform=None,
        pathcol="full_path",
        data_path="",
        fs=16000,
        percent=1,
        time_dim=0,
    ):
        """
        Class for loading audio data with quality (or any target) scores.

        Parameters
        ----------
        data_files : list
            List of paths to data files containing audio paths and target scores.
        transform : _type_, optional
            Transform object for applying transforms to audio, by default None
        transform_time : str, optional
            When to perform transforms, by default "get"
        cache : bool, optional
            Store audio rather than loading it each time, by default False
        target : str, optional
            Target to train to, must be column name in data files, by default "MOS"
        pathcol : str, optional
            Name of column containing audio path in data files, by default "full_path"
        data_path : str, optional
            Parent path to data for paths stored in pathcol, by default "".
        fs : int, optional
            Sampling rate required for the network. All audio will be resampled
            to this sample rate if necessary. If fs is set to None, the resampling
            step will be skipped.
        percent : float, optional
            Percentage of data to use. Data will be randomly partitioned according
            to percent. Primarily intended for debugging.
        """
        self.fs = fs
        self.percent = percent
        self.time_dim = time_dim

        score_files = list()
        # Load each data file
        for k, file in enumerate(data_files):
            score_file = pd.read_csv(file)
            if self.percent < 1:
                n_items = len(score_file)
                keep_items = np.ceil(n_items * self.percent).astype(int)
                rng = np.random.default_rng()
                keep_ix = rng.choice(len(score_file), keep_items, replace=False)
                score_file = score_file.loc[keep_ix]
            score_file["Dataset_Indicator"] = k
            score_files.append(score_file)

        self.score_file = pd.concat(score_files, ignore_index=True)
        # Initialize dictionary to store audio in if cache is True
        if cache:
            self.wavs = dict()

        self.data_path = data_path
        self.pathcol = pathcol
        self.target = target
        self.target_transform = target_transform
        self.transform = transform
        self.transform_time = transform_time

    def __len__(self):
        return len(self.score_file)

    def __getitem__(self, idx):
        dataset = self.score_file.loc[idx, "Dataset_Indicator"]
        if self.target is not None:
            mos = self.score_file.loc[idx, self.target]
        else:
            mos = None
        if self.target_transform is not None:
            mos = self.target_transform(mos, dataset)

        audio_path = os.path.join(
            self.data_path, self.score_file.loc[idx, self.pathcol]
        )
        if hasattr(self, "wavs") and audio_path in self.wavs:
            # Already loaded and cached the audio
            audio = self.wavs[audio_path]
        else:
            # Load audio
            audio, sample_rate = torchaudio.load(audio_path)

            # Resample as needed
            if self.fs is not None and sample_rate != self.fs:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, self.fs, dtype=audio.dtype
                )
                audio = resampler(audio)
                sample_rate = self.fs

            if self.transform is not None and self.transform_time == "get":
                # Apply transform

                if self.fs is None:
                    # We are not resampling which means the transform should handle this
                    audio = self.transform.transform(audio, fs=sample_rate)
                else:
                    # We have resampled, so the transform doesn't need to account for it
                    audio = self.transform.transform(audio)

            audio = audio.float()
            if hasattr(self, "wavs"):
                self.wavs[audio_path] = audio
        return audio, mos, dataset

    def padding(self, batch):
        """
        Pad inputs in a batch so that they have the same dimensions.

        Parameters
        ----------
        batch : tuple
            Tuple of items in the batch.

        Returns
        -------
        tuple
            Batch with padding applied so that all of audio is same dimension.
        """
        # Unpack data within batch
        audio_files, mos, dataset = zip(*batch)

        mos = torch.tensor(np.array(mos))
        dataset = torch.tensor(dataset)

        # Find maximum length in time dimension
        max_len = np.max([audio.shape[self.time_dim] for audio in audio_files])

        audio_out = []
        for ix, audio in enumerate(audio_files):
            if audio.shape[self.time_dim] < max_len:
                repeat_samples = max_len - audio.shape[self.time_dim]
                # Initialize pad width for each dimension to no padding
                pad_width = [(0, 0) for i in range(len(audio.shape))]
                # Dimension always time - pad end with repeat_samples
                pad_width[self.time_dim] = (0, repeat_samples)
                # Convert to tuple for input to np.pad
                pad_width = tuple(pad_width)

                audio = np.pad(audio, pad_width=pad_width, mode="constant")
                audio = torch.from_numpy(audio)
            audio_out.append(audio)

        # Concatenate into one tensor
        audio_out = torch.stack(audio_out, dim=0)
        # If a transform is defined and the transform time is at collate now is the time to apply it
        if self.transform is not None and self.transform_time == "collate":
            audio_out = self.transform.transform(audio_out)
        audio_out = torch.unsqueeze(audio_out, dim=1)
        return audio_out, mos, dataset


class FeatureData(AudioData):
    """
    For loading pre-computed features for audio files. Only the __getitem__ method needs to change
    """

    def __init__(
        self,
        flatten=True,
        dim_cutoff=None,
        dim=0,
        **kwargs,
    ):
        """
        Class for pre-computed features of audio files.

        Inherits from AudioData.

        Parameters
        ----------
        flatten : bool, optional
            Flatten representation into a single dimension, by default True
        dim_cutoff : _type_, optional
            Max number of dimensions to consider. By default None.
        dim : int, optional
            Dimension on which to perform cutoff using dim_cutoff, by default 0
        """
        super().__init__(
            **kwargs,
        )
        self.dim_cutoff = dim_cutoff
        self.dim = dim
        self.flatten = flatten

    def __getitem__(self, idx):
        dataset = self.score_file.loc[idx, "Dataset_Indicator"]
        
        if self.target is not None:
            mos = self.score_file.loc[idx, self.target]
        else:
            mos = None
        
        if self.target_transform is not None:
            mos = self.target_transform(mos, int(dataset))

        audio_path = os.path.join(
            self.data_path, self.score_file.loc[idx, self.pathcol]
        )
        if hasattr(self, "wavs") and audio_path in self.wavs:
            # Already loaded and cached the audio
            audio = self.wavs[audio_path]
        else:
            fname, ext = os.path.splitext(audio_path)
            # If using same split csvs as audio this may say wav and not pt
            # (coming out of pretransform_data.py will save as pt)
            if ext == ".wav":
                audio_path = fname + ".pkl"
            # Load audio
            with open(audio_path, "rb") as feat_input:
                audio = pickle.load(feat_input)

            if self.dim_cutoff is not None:
                audio = torch.narrow(
                    audio, dim=self.dim, start=0, length=self.dim_cutoff
                )

            if self.flatten:
                # Flatten by column
                audio = audio.t().flatten()

            if self.transform is not None and self.transform_time == "get":
                # Apply transform
                audio = self.transform.transform(audio)

            audio = audio.float()
            if hasattr(self, "wavs"):
                self.wavs[audio_path] = audio

        return audio, mos, dataset


class AudioDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dirs,
        batch_size=16,
        num_workers=1,
        persistent_workers=True,
        DataClass=AudioData,
        collate_type="padding",
        data_percent=1,
        **kwargs,
    ):
        """
        Primary audio data module that prepares data for training, testing, or predictions.

        Parameters
        ----------
        data_dirs : list
            List of paths to directories containing train.csv, valid.csv, and test.csv
            for each dataset.
        batch_size : int, optional
            Number of items in each batch, by default 32
        num_workers : int, optional
            Number of workers used during training, by default 1
        persistent_workers : bool, optional
            Whether or not workers persist between epochs, by default True
        DataClass : class, optional
            Class that the data will be initialized with. Assumed to inherit from
            torch.utils.data.Dataset, by default AudioData
        collate_type : str, optional
            String that determines what type of collate function is used, by default
            "padding"
        **kwargs : optional
            Additional arguments are passed to the DataClass when instantiated in 
            AudioDataModule.setup()

        """
        super().__init__()

        # If this class sees batch_size=auto it sets to default value and assumes a Tuner is being called in the main
        # logic to update this later
        if batch_size == "auto":
            batch_size = 32
        self.batch_size = batch_size
        self.collate_type = collate_type
        self.data_dirs = data_dirs
        self.DataClass = DataClass
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

        self.data_class_kwargs = kwargs

    def setup(self, stage: str):
        """
        Load different datasubsets depending on stage.

        If stage == 'fit' then train, valid, and test data are loaded.

        If stage == 'test' then only test data is loaded.

        If stage == 'predict' then self.data_dirs should be full paths to the specific 
        csv files to run predictions on.

        Parameters
        ----------
        stage : str
            One of fit, test, or predict.
        """
        if stage == "fit":
            train_paths = self.find_datasubsets(self.data_dirs, "train")
            self.train = self.DataClass(
                data_files=train_paths,
                **self.data_class_kwargs,
            )

            valid_paths = self.find_datasubsets(self.data_dirs, "valid")
            self.valid = self.DataClass(
                data_files=valid_paths,
                **self.data_class_kwargs,
            )

            test_paths = self.find_datasubsets(self.data_dirs, "test")
            self.test = self.DataClass(
                data_files=test_paths,
                **self.data_class_kwargs,
            )
        elif stage == "test":
            test_paths = self.find_datasubsets(self.data_dirs, "test")
            self.test = self.DataClass(
                data_files=test_paths,
                **self.data_class_kwargs,
            )
        elif stage == "predict":
            self.predict = self.DataClass(
                data_files=self.data_dirs,
                **self.data_class_kwargs,
            )

    def find_datasubsets(self, data_paths, subset):
        """
        Find subsets as determined by data_paths and subset.

        Primarily relies on find_datasubset. Loads in train.csv, valid.csv, and
        test.csv as necessary.

        Parameters
        ----------
        data_paths : list
            List of paths for each dataset.
        subset : str
            Which type of csv file to read in.

        Returns
        -------
        list
            Paths to that datasubset.
        """
        outs = []
        for data_path in data_paths:
            out = self.find_datasubset(data_path, subset)
            outs.append(out)
        return outs

    def find_datasubset(self, data_path, subset):
        """
        Helper function for setup to find the different data subsets (test/train/valid)

        Parameters
        ----------
        data_path : str
            Path to folder containg subset.csv
        subset : str
            String for representative datasubset file in data_path

        Returns
        -------
        str
            path to subset.csv

        Raises
        ------
        ValueError
            subset could not be found in data_path.
        """
        _, ext = os.path.splitext(data_path)
        if ext == ".csv":
            return data_path
        files = os.listdir(data_path)
        out = []
        for file in files:
            basename, ext = os.path.splitext(file)
            if basename == subset:
                out.append(file)
        if len(out) == 0:
            raise ValueError(f"Unable to find {subset} in {data_path}")
        elif len(out) > 1:
            warnings.warn(
                f"Multiple matches for {subset} in {data_path}.\nOf {out} using {out[0]}"
            )
            out = out[0]
        else:
            out = out[0]
        out = os.path.join(data_path, out)
        return out

    def train_dataloader(self):
        """
        Prepare DataLoader for training.
        """
        if self.collate_type == "padding":
            collate_fn = self.train.padding
        else:
            collate_fn = None
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        """
        Prepare DataLoader for validation.
        """
        if self.collate_type == "padding":
            collate_fn = self.valid.padding
        else:
            collate_fn = None
        return DataLoader(
            self.valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        """
        Prepare DataLoader for testing.
        """
        if self.collate_type == "padding":
            collate_fn = self.test.padding
        else:
            collate_fn = None
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self):
        """
        Prepare DataLoader for prediction.
        """
        if self.collate_type == "padding":
            collate_fn = self.predict.padding
        else:
            collate_fn = None
        return DataLoader(
            self.predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_fn,
        )
