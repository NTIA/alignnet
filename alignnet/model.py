from typing import Any
import hydra
import os
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torchaudio
import yaml
import warnings

import numpy as np
import pytorch_lightning as pl

from omegaconf import DictConfig
from torch import nn

from .optimizer import OptimizerWrapper
from .transforms import MelTransform


def load_model(trained_model_path):
    """
    Load a model directory.

    Parameters
    ----------
    trained_model_path : str
        Path to trained_model directory containing a config.yaml and model.ckpt
        required to load a pretrained model.

    Returns
    -------
    Model
        Pretrained alignnet.Model object.
    """
    # Load config
    cfg_path = os.path.join(trained_model_path, "config.yaml")
    with open(cfg_path, "r") as f:
        cfg_yaml = yaml.safe_load(f)
    cfg = DictConfig(cfg_yaml)
    # Initialize network
    network = hydra.utils.instantiate(cfg.network)

    model_class = hydra.utils.get_class(cfg.model._target_)

    model_path = os.path.join(trained_model_path, "model.ckpt")
    # Initialize model
    model = model_class.load_from_checkpoint(model_path, network=network)

    return model


def mean_pooling(frame_scores, dim=1):
    """
    Time pooling method that averages frames.

    Parameters
    ----------
    frame_scores : torch.tensor
        Frame-wise estimates.
    dim : int, optional
        Dimension along which to average, by default 1

    Returns
    -------
    torch.tensor
        Frame averaged estimates.
    """
    mean_estimate = torch.mean(frame_scores, dim=dim)
    mean_estimate = torch.squeeze(mean_estimate)
    return mean_estimate


# ------------------------
# Audio Processing Modules
# ------------------------


class LinearSequence(nn.Module):
    def __init__(
        self,
        in_features,
        n_layers=2,
        activation=nn.ReLU,
        layer_dims=None,
        last_activate=False,
    ):
        """
        Generate a sequence of n_layers Fully Connected (nn.linear) layers with activation.

        Parameters
        ----------
        n_layers : int
            Number of linear layers to include.
        in_features : int
            Number of features in input
        activation : nn.Module
            Activation to include between layers. There will always be n_layers - 1 activations in the sequence.
        layer_dims : list
            List of layer dimensions, not including input features (these are specificed by in_features).
        """
        # TODO - the inputs for this are poorly defined/redundant. A refactor could really improve this/make more generalizable.
        super().__init__()
        if layer_dims is not None and n_layers != len(layer_dims):
            n_layers = len(layer_dims)
        self.n_layers = n_layers
        self.in_features = in_features
        self.activation = activation
        self.layer_dims = layer_dims
        self.last_activate = last_activate
        self.setup_layers(self.n_layers)

    def setup_layers(self, n_layers):
        """
        Setup and store layers into `output_layers` attribute.

        If `self.layer_dims` is not None, linear layers are made that match the
        dimension of that list. If it is none, layers are made such that the dimension
        decreases by 1/2 for each layer.

        Parameters
        ----------
        n_layers : int
            Number of layers to make if `self.layer_dims` is not defined.
        """
        n_features = self.in_features
        layers = []

        if self.layer_dims is None:
            for k in range(n_layers - 1):
                next_feat = int(n_features / 2)
                layer = nn.Linear(n_features, next_feat)
                layers.append(layer)
                layers.append(self.activation())
                n_features = next_feat
            # Final layer to map to MOS
            layers.append(nn.Linear(n_features, 1))
        else:
            for k, layer_dim in enumerate(self.layer_dims):
                # Map previous number of features to layer dim
                layer = nn.Linear(n_features, layer_dim)
                layers.append(layer)
                if k < len(self.layer_dims) - 1:
                    layers.append(self.activation())
                elif k == len(self.layer_dims) - 1 and self.last_activate:
                    layers.append(self.activation())

                # Save current layer dim as previous number of features
                n_features = layer_dim

        self.output_layers = nn.ModuleList(layers)

    def forward(self, frame_scores):
        """
        Forward method for fully connected linear sequence.

        Parameters
        ----------
        frame_scores : torch.tensor
            Input tensor for linear sequence.

        Returns
        -------
        torch.Tensor
            Frame based representation of audio (e.g. feature x frames tensor for each audio file).
        """
        for k, layer in enumerate(self.output_layers):
            frame_scores = layer(frame_scores)
        return frame_scores


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.3):
        """
        Convolutional block used in MOSNet.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        dropout : float, optional
            Dropout probability, by default 0.3
        """
        super().__init__()
        self.block = nn.Sequential(
            # Input shape: (B, T, in_channels, N)
            # Output shape: (B, T, out_channels, ceil(N/3)) (stride 3 in freq of last convolutional layer causes decrease)
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 3),
                padding=(1, 1),
            ),
            nn.Dropout(0.3),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class MOSNet(nn.Module):
    def __init__(
        self,
    ):
        """
        Implement MOSNet architecture, mostly as described in "MOSNet: Deep Learning based
        Objective Assessment for Voice Conversion" by Lo et al. (2019).

        Unlike the original, this implementation does not implement frame-level loss.
        """
        super().__init__()
        self.convolutions = nn.Sequential(
            # Input shape: (B, 1, T, 257)
            # Output shape: (B, 16, T, 86) (stride 3 in freq of last convolutional layer causes decrease)
            ConvBlock(1, 16),
            # Input shape: (B, 16, T, 86)
            # Output shape: (B, 32, T, 29)
            ConvBlock(16, 32),
            # Input shape: (B, 32, T, 29)
            # Output shape: (B, 64, T, 10)
            ConvBlock(32, 64),
            # Input shape: (B, 64, T, 10)
            # Output shape: (B, 128, T, 4)
            ConvBlock(64, 128),
        )
        self.blstm = nn.LSTM(
            # input_size - number of expected features in input x
            input_size=512,  # 4*128
            # hidden_size - number of features in hidden state h
            hidden_size=128,
            # num_layers - number of recurrent layers
            num_layers=1,
            # bias - bool if bias weight used (defaults to True)
            # batch_first - if True then input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)
            batch_first=True,
            # dropout
            # bidirectional
            bidirectional=True,
        )  # (B, T, 256=2*128), 2 b/c bidirectional==True, 128 b/c hidden_size=128 and proj_size=0
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=256,
                out_features=128,
            ),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(
                in_features=128,
                out_features=1,
            ),
        )
        self.time_pooling = torch.mean

    def forward(self, x):
        # x dim: (B, C, T, F) = (B, T, 1, 257)
        y = self.convolutions(x)
        # y dim: (B, C, T, F) = (B, 128, T, 4)
        # Swap dimensions to preserve frame level time before flattening for BLSTM
        y = torch.movedim(y, -2, -3).flatten(start_dim=-2, end_dim=-1)
        # y dim: (B, T, F*C): (B, T, 512)
        y, _ = self.blstm(y)
        # y dim: (B, T, 2*H): (B, T, 256) -- H is hidden dimension, 2x b/c Bidirectional
        y = self.fc(y)
        # y dim: (B, T, 1)
        y = self.time_pooling(y, dim=1)
        # y dim: (B, 1)
        # y = torch.squeeze(y)
        # # y dim: B
        return y


class AudioConvolutionalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        padding=0,
        pooling_type="average",
        pooling_kernel=4,
        batch_norm=True,
    ):
        """
        Audio convolutional block in the style of WAWENets.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int
            Size of convolution kernel.
        stride : int, optional
            Convolution stride, by default 1
        dilation : int, optional
            Convolution dilation, by default 1
        pooling_type : str, optional
            Type of pooling to perform. Either "average", "blur", or "None", by default "average".

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.pooling_type = pooling_type
        self.pooling_kernel = pooling_kernel
        self.batch_norm = batch_norm

        self.setup()

    def setup(self):
        """
        Set up the modules.

        Sets up a convolution, batch norm (optional), ReLU, and pooling (optional).
        """
        model_list = []
        conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
        )
        model_list.append(conv)
        # TODO - figure out what the batchnorm num_features needs to be
        if self.batch_norm:
            bn = nn.BatchNorm1d(self.out_channels)
            model_list.append(bn)

        relu = nn.ReLU()
        model_list.append(relu)

        if self.pooling_type is not None:
            if self.pooling_type == "average":
                pool = nn.AvgPool1d(kernel_size=self.pooling_kernel)
            elif self.pooling_type == "blur":
                raise ValueError(f"blur pooling not implemented yet.")
            else:
                raise ValueError(f"Unrecognized pooling_type `{self.pooling_type}`")
            # add pool to the nn.ModuleList
            model_list.append(pool)
        self.model_list = nn.ModuleList(model_list)

    def forward(self, x):
        for module in self.model_list:
            x = module(x)
        return x


class MuLaw(nn.Module):
    def __init__(self):
        """
        Learnable Mu-Law compression module.
        """
        super().__init__()
        self.mu = torch.nn.Parameter(torch.tensor([255.0]))

    def forward(self, x):
        sign = torch.sign(x)
        out = sign * torch.log(1 + self.mu * torch.abs(x)) / torch.log(1 + self.mu)
        return out


class ConvPath(nn.Module):
    def __init__(
        self,
        kernels,
        strides,
        dilations,
        channels,
        paddings,
        pooling_kernels=[None],
        in_channel=1,
        rectify=False,
        mu_law=False,
        **kwargs,
    ):
        """
        Convolutional paths for multi-scale convolution.

        Parameters
        ----------
        kernels : list
            List of kernel sizes within the path. The length of kernels determines
            the number of elements in the convolutional path.
        strides : list
            List of strides within the path. Can be one element list and will be repeated
            to be the same length as kernels.
        dilations : list
            List of dilations within the path. Can be one element list and will be repeated
            to be the same length as kernels.
        channels : list
            List of channels within the path. Can be one element list and will be repeated
            to be the same length as kernels.
        paddings : list
            List of paddings within the path. Can be one element list and will be repeated
            to be the same length as kernels.
        pooling_kernels : list, optional
            List of poolings within the path. Can be one element list and will be repeated
            to be same length as kernels, by default [None]
        in_channel : int, optional
            Number of channels in first AudioConvolutionalBlock, by default 1
        rectify : bool, optional
            Rectify signal at beginning of the path, by default False
        mu_law : bool, optional
            Apply learnable Mu-Law compression at beginning of path, by default False
        """
        super().__init__()
        n_blocks = len(kernels)

        # make sure any single element lists have same length as kernels
        if len(strides) == 1:
            strides = n_blocks * list(strides)
        if len(dilations) == 1:
            dilations = n_blocks * list(dilations)
        if len(channels) == 1:
            channels = n_blocks * list(channels)
        if len(paddings) == 1:
            paddings = n_blocks * list(paddings)
        if len(pooling_kernels) == 1:
            pooling_kernels = n_blocks * list(pooling_kernels)

        self.rectify = rectify
        self.mu_law = mu_law
        conv_blocks = []
        if mu_law:
            mu = MuLaw()
            conv_blocks.append(mu)

        for ix, (
            kernel_size,
            stride,
            dilation,
            channel,
            padding,
            pooling_kernel,
        ) in enumerate(
            zip(kernels, strides, dilations, channels, paddings, pooling_kernels)
        ):
            if ix > 0:
                in_channel = channel
            # Initialize conv_block
            conv_block = AudioConvolutionalBlock(
                in_channels=in_channel,
                out_channels=channel,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                pooling_kernel=pooling_kernel,
            )

            # Set any additional AudioConvolutionalBlock parameters that have been passed in through kwargs
            for k, v in kwargs.items():
                if hasattr(conv_block, k):
                    setattr(conv_block, k, v)
            # Reconfigure modules
            conv_block.setup()
            conv_blocks.append(conv_block)
        self.conv_blocks = nn.ModuleList(conv_blocks)

    def forward(self, x):
        if self.rectify:
            x = torch.abs(x)

        for conv_block in self.conv_blocks:
            x = conv_block(x)
        return x


class IdentityBlock(nn.Module):
    def __init__(self, **kwargs):
        """
        Identity block for audio-style processing that returns the input in the forward method.
        """
        super().__init__()

    def forward(x):
        return x


class MultiScaleConvolution(nn.Module):
    def __init__(self, path1, path2, path3, path4):
        """
        Neural network that processes audio in up to four independent paths prior to
        combining in a fully connected sequence. Each path is compressed to be the same
        size regardless of audio length through simple statistical aggregations.

        Parameters
        ----------
        path1 : nn.Module
            First of the four independent paths. Will be ignored if set to IdentityBlock.
        path2 : nn.Module
            Second of the four independent paths. Will be ignored if set to IdentityBlock.
        path3 : nn.Module
            Third of the four independent paths. Will be ignored if set to IdentityBlock.
        path4 : nn.Module
            Fourth of the four independent paths. Will be ignored if set to IdentityBlock.
        """
        super().__init__()
        paths = [path1, path2, path3, path4]

        # Drop any identity paths.
        paths = [path for path in paths if not isinstance(path, IdentityBlock)]

        self.conv_paths = nn.ModuleList(paths)

        # Track total dimension of convolutional outputs from all the paths
        conv_out_dimension = 0
        for path in self.conv_paths:
            out_dim = path.conv_blocks[-1].model_list[0].out_channels
            conv_out_dimension += 2 * out_dim

        # Sequence of fully connected layers
        self.decoder = LinearSequence(
            in_features=conv_out_dimension,
            layer_dims=[int(conv_out_dimension / 4), int(conv_out_dimension / 16), 1],
        )

    def forward(self, x):
        if len(x.shape) > 3 and x.shape[1] == 1:
            # This may not be the best place/way to do this but should work on mono audio
            x = torch.squeeze(x, dim=1)
        path_outs = []
        for conv_path in self.conv_paths:
            conv_out = conv_path(x)

            conv_means = torch.mean(conv_out, dim=-1)
            conv_stds = torch.std(conv_out, dim=-1)
            conv_stat = torch.cat((conv_means, conv_stds), dim=-1)
            path_outs.append(conv_stat)

        stats = torch.cat(path_outs, -1)

        out = self.decoder(stats)
        return out


# -----------------------
# Aligner - Dataset Alignment Modules
# -----------------------
class NoAligner(nn.Module):
    def __init__(self, reference_index=0, num_datasets=0, **kwargs):
        """
        NoAligner acts as a dummy module so that the other AlignNet module code
        can be used even when there is no dataset alignment being performed.

        Parameters
        ----------
        reference_index : int, optional
            Unused but exists to easily replace other Aligner setups, by default 0
        num_datasets : int, optional
            Unused but exists to easily replace other Aligner setups, by default 0
        """
        super().__init__()
        self.reference_index = reference_index
        self.num_datasets = num_datasets

    def forward(self, stilde, dataset_index):
        return stilde


class LinearSequenceAligner(nn.Module):
    def __init__(
        self,
        reference_index,
        num_datasets,
        embedding_dim=10,
        layer_dims=[16, 32, 16, 1],
    ):
        """
        Aligner network for dataset alignment.

        The LinearSequenceAligner implements the Aligner as defined in "AlignNet:
        Learning dataset score alignment functions to enable better training of
        speech quality estimators."

        Parameters
        ----------
        reference_index : int
            Dataset index that should be treated as the reference. The Aligner acts
            as the identity function on the reference dataset.
        num_datasets : int
            Number of datasets in training.
        embedding_dim : int, optional
            Size of the dataset index embedding, by default 10
        layer_dims : list, optional
            Dimensions of the Aligner fully connected layers, by default [16, 32, 16, 1]
        """
        super().__init__()
        self.reference_index = reference_index
        self.num_datasets = num_datasets
        # TODO - this should probs be smart
        self.embedding_dim = embedding_dim
        self.embedding = torch.nn.Embedding(
            num_datasets, embedding_dim=self.embedding_dim
        )
        self.weights = LinearSequence(
            in_features=self.embedding_dim + 1,
            layer_dims=layer_dims,
        )

    def forward(self, stilde, dataset_index):
        embed = self.embedding(dataset_index)
        # Concatenate score with embedding
        x = torch.cat((stilde, embed), 1)
        score = self.weights(x)
        # Override for reference dataset -- this might be weird!
        out = torch.where(dataset_index[:, None] == self.reference_index, stilde, score)
        return out


# ------------------
# Combination Module
# ------------------
class AlignNet(nn.Module):
    def __init__(
        self, audio_net, aligner, aligner_corr_threshold=None, audio_net_freeze_epochs=0
    ):
        """
        AlignNet module that uses an audio_net with an aligner to train on multiple datasets at once.

        Parameters
        ----------
        audio_net : nn.Module
            Network component that maps audio to quality on the reference dataset scale.
        aligner : nn.Module
            Network componenent that maps intermediate quality estimates and dataset
            indicators to the appropriate dataset score.
        aligner_corr_threshold : float, optional
            Correlation threshold that determines when the aligner is activated.
            If None the aligner turns on immediately, by default None
        audio_net_freeze_epochs : int, optional
            Number of epochs to keep the audio_net frozen, by default 0
        """
        super().__init__()
        self.audio_net = audio_net
        self.aligner = aligner
        self.reference_index = self.aligner.reference_index

        if aligner_corr_threshold is not None:
            # We want to freeze aligner (and ideally ensure it is not changing
            # estimations) until we see a validation correlation above
            # aligner_corr_threshold.
            self.use_aligner = False
            self.aligner_corr_threshold = aligner_corr_threshold

            # Freeze aligner params
            for p in self.aligner.parameters():
                p.requires_grad_(False)

        else:
            self.use_aligner = True
            self.aligner_corr_threshold = -1

        self.audio_net_freeze_epochs = audio_net_freeze_epochs

        if audio_net_freeze_epochs > 0:
            self.set_audio_net_update_status(False)
        else:
            self.update_audio_net = True

    def set_audio_net_update_status(self, status):
        self.update_audio_net = status
        for p in self.audio_net.parameters():
            p.requires_grad_(status)

    def forward(self, audio, dataset):
        # Intermediate score representation
        score = self.audio_net(audio)
        if self.use_aligner:
            # Aligned score estimate
            score = self.aligner(score, dataset)
        return score


# --------------
# Primary Module
# --------------
class Model(pl.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        loss=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam,
        loss_weights=None,
    ):
        """
        LightningModule to train AlignNet models.

        Module should be compatible with non-AlignNet architecture but includes
        additional functionality specifically tailored to AlignNet.

        Parameters
        ----------
        network : nn.Module
            AlignNet model.
        loss : func, optional
            Loss function, by default torch.nn.MSELoss()
        optimizer : OptimizerWrapper or torch.nn.optim class, optional
            Optimizer class, by default torch.optim.Adam
        loss_weights : list
            List of weights to compute weighted average of loss over datasets. If None, then loss is computed without
            respect to datasets. In the case where one dataset has significantly less data, a weighted average allows
            more control to ensure it is properly learned. If loss_weights = 1, then the all datasets will get equal weight.
        """
        super().__init__()
        # self.save_hyperparameters(ignore=["network", "loss"])
        self.network = network

        self.loss = loss
        if loss_weights == 1:
            n_datasets = self.network.aligner.num_datasets
            loss_weights = n_datasets * [1 / n_datasets]
        self.loss_weights = loss_weights
        self.optimizer = optimizer
        self.validation_step_info = {
            "outputs": [],
            "targets": [],
            "datasets": [],
        }
        self.epoch = 0

    def loss_calc(self, mean_estimate, mos, dataset):
        """
        Perform loss calculation, taking into account loss weights.

        Parameters
        ----------
        mean_estimate : torch.tensor
            Network estimate.
        mos : torch.tensor
            Labeled truth value.
        dataset : torch.tensor
            Dataset indicators.

        Returns
        -------
        torch.tensor
            Loss.
        """
        # If there are loss weights use them
        if self.loss_weights is not None:
            loss = 0
            for dix in torch.unique(dataset):
                dix = int(dix)
                weight = self.loss_weights[dix]
                sub_ests = mean_estimate[dataset == dix]
                sub_mos = mos[dataset == dix]
                loss += weight * self.loss(sub_ests, sub_mos)

        else:
            loss = self.loss(mos, mean_estimate)
        return loss

    def forward(self, audio, dataset):
        mean_estimate = self.network(audio, dataset)
        mean_estimate = torch.squeeze(mean_estimate, dim=1)
        return mean_estimate

    def _forward(self, training_batch):
        """
        Internal forward method with logic consistent across all training and test steps.

        Parameters
        ----------
        training_batch : tuple
            All data in a training batch.
        """
        audio, mos, dataset = training_batch
        mos = mos.float()  # TODO - not sure this is the right place for this
        # if self._shift_mean:
        #     audio = audio - self.mean

        mean_estimate = self.network(audio, dataset)
        # If audio is 2-D (e.g. wav2vec representation) needs to be squeezed in diminsion 1 here
        # If audio is raw wav this won't do anything (dim 1 will be frames and != 1)
        mean_estimate = torch.squeeze(mean_estimate, dim=1)

        loss = self.loss_calc(mean_estimate, mos, dataset)

        if mos.shape == torch.Size([1]):
            warnings.warn(f"Batch only has one element, reporting correlation=0")
            corrcoef = 0
        else:
            corrcoef = self.pearsons_corr(mean_estimate, mos)
        return loss, corrcoef

    def pearsons_corr(self, mean_estimate, mos):
        """
        Simple wrapper for grabbing pearsons correlation coefficient
        """
        mean_estimate = torch.unsqueeze(mean_estimate, dim=1)
        mos = torch.unsqueeze(mos, dim=1)

        cat = torch.cat([mean_estimate, mos], dim=1)
        cat = torch.transpose(cat, 0, 1)

        corrcoef = torch.corrcoef(cat)[0, 1]

        return corrcoef

    def training_step(self, training_batch, idx):
        loss, corrcoef = self._forward(training_batch)
        self.log("train_loss", loss)
        self.log("train_pearsons", corrcoef)
        return loss

    def validation_step(self, val_batch, idx):
        """
        Validtion step. Unlike the training and test steps, we need to store per
        dataset information here.
        """
        audio, mos, dataset = val_batch
        mos = mos.float()

        mean_estimate = self.network(audio, dataset)
        mean_estimate = torch.squeeze(mean_estimate, dim=1)

        loss = self.loss_calc(mean_estimate, mos, dataset)

        # Store per dataset information to be used at epoch end.
        self.validation_step_info["outputs"].append(mean_estimate)
        self.validation_step_info["targets"].append(mos)
        self.validation_step_info["datasets"].append(dataset)

        return loss

    def on_validation_epoch_end(self) -> None:
        """
        At the end of validation epochs we calculate per dataset statistics.
        """
        # Concatenate stored epoch data into single tensor for each metric
        estimates = torch.cat(self.validation_step_info["outputs"], dim=0)
        targets = torch.cat(self.validation_step_info["targets"], dim=0)
        datasets = torch.cat(self.validation_step_info["datasets"], dim=0)

        # Overall loss and correlatoin
        loss = self.loss_calc(estimates, targets, datasets)
        corrcoef = self.pearsons_corr(estimates, targets)

        # Check if network has a use_aligner flag
        if hasattr(self.network, "use_aligner"):
            # If aligner is off and we have passed the correlation threshold do the updates
            if (
                not self.network.use_aligner
                and corrcoef > self.network.aligner_corr_threshold
            ):
                # Start using alignment network in forward
                self.network.use_aligner = True
                # Turn on gradients for aligner parameters
                for p in self.network.aligner.parameters():
                    p.requires_grad_(True)
                print(
                    f"Correlation threshold of {self.network.aligner_corr_threshold} reached with {corrcoef:.4f}. Turning on aligner."
                )

        self.log("val_loss", loss)
        self.log("val_pearsons", corrcoef)

        # Per dataset losses and correlations
        for k, ds in enumerate(torch.unique(datasets)):
            ds_ix = datasets == ds

            ds_est = estimates[ds_ix]
            ds_tar = targets[ds_ix]

            ds_loss = self.loss(ds_est, ds_tar)
            ds_corr = self.pearsons_corr(ds_est, ds_tar)
            self.log(f"val_loss/dataset {k}", ds_loss)
            self.log(f"val_pearsons/dataset {k}", ds_corr)

        # Clear epoch validation info
        for k, v in self.validation_step_info.items():
            v.clear()

        # If we aren't updating audio-net and our epoch has passed the wait time turn it on!
        if (
            not self.network.update_audio_net
            and self.epoch >= self.network.audio_net_freeze_epochs
        ):
            # Turn on audio_net, set
            print(f"Turning audio_net on after {self.epoch} epochs.")
            self.network.set_audio_net_update_status(True)
        self.epoch += 1

        return super().on_validation_epoch_end()

    def test_step(self, test_batch, idx):
        loss, corrcoef = self._forward(test_batch)
        self.log("test_loss", loss)
        self.log("test_pearsons", corrcoef)
        return loss

    def configure_optimizers(self):
        if isinstance(self.optimizer, OptimizerWrapper):
            optimizer = self.optimizer.optimizer(self.parameters())
        else:
            optimizer = self.optimizer(self.parameters())
        return optimizer
