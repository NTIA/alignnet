import contextlib
from typing import Any
import fms
import torch
import torchaudio

import numpy as np


# ------------------
# Audio Transforms
# -----------------
class NoneTransform:
    def __init__(self):
        pass

    def transform(self, audio):
        return audio


class MelTransform:
    def __init__(self, fft_win_length, win_overlap, n_mels):
        """
        Mel Spectrogram transform.

        Parameters
        ----------
        sample_rate : int
            Sample rate of audio.
        fft_win_length : int
            Window length, in samples.
        win_overlap : int
            Window overlap, in samples.

        """
        self.win_length = win_length
        self.win_overlap = win_overlap
        self.n_mels = n_mels

    def transform(self, audio, sample_rate, n_mels=None, device="cpu"):
        """
        Perform mel spectrogram transform

        Parameters
        ----------
        audio : torch.tensor
            Audio to transform.
        sample_rate : int
            Sample rate of audio.
        n_mels : int, optional
            Number of mel bands in transform, by default 32.
        device : torch.device
            Device to perform transform on, by default "cpu".

        Returns
        -------
        torch.tensor
            Transformed audio.
        """

        if n_mels is None:
            n_mels = self.n_mels

        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.win_length,
            # win_length=self.win_length,
            hop_length=self.win_length - self.win_overlap,
            n_mels=n_mels,
            center=False,
            norm="slaney",
            mel_scale="slaney",
        )
        transform = transform.to(device)

        mel = transform(audio)
        mel = torch.squeeze(mel)

        return mel


class STFTTransform:
    def __init__(self, fft_win_length=512, win_overlap=256):
        """
        Short-time fourier transform.

        Parameters
        ----------
        fft_win_length : int, optional
            Window length in samples, by default 512
        win_overlap : int, optional
            Window overlap in samples, by default 256
        """
        self.fft_win_length = fft_win_length
        self.win_overlap = win_overlap

    def transform(self, audio):
        """
        Perform a STFT on audio.

        Parameters
        ----------
        audio : torch.tensor
            Audio to transform.

        Returns
        -------
        torch.tensor
            Transformed audio.
        """
        hann_window = torch.hann_window(
            window_length=self.fft_win_length,
            periodic=True,
        )

        stft = torch.stft(
            audio,
            n_fft=self.fft_win_length,
            hop_length=self.fft_win_length - self.win_overlap,
            window=hann_window,
            return_complex=True,
            center=False,
        )
        stft = torch.abs(stft)  # (N_freq, N_frame)
        stft = torch.movedim(stft, -1, -2)  # (N_frame, N_freq)
        stft = torch.squeeze(stft)  # (N_frame, N_freq)

        return stft
