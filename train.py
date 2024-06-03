import hydra
import os
import shutil
import torch
import warnings
import yaml

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm
import alignnet

# Load clearml Task only if clearml is imported
try:
    from clearml import Task
except ModuleNotFoundError as err:

    def Task(**kwargs):
        return None


def post_train(model, audio_data, loggers, task=None):
    audio_data.batch_size = 1
    data_loaders = {
        "train": audio_data.train_dataloader(),
        "val": audio_data.val_dataloader(),
        "test": audio_data.test_dataloader(),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Save estimations
    out_df = pd.DataFrame()
    with torch.no_grad():
        for dataset_split, loader in data_loaders.items():
            print(f"Loading dataset: {dataset_split}")
            results = []
            for audio, mos, dataset in tqdm(loader):
                audio = audio.to(device)
                dataset = dataset.to(device)

                est = model(audio, dataset)
                est = est.numpy(force=True)
                est = np.squeeze(est)

                audio_net_est = model.network.audio_net(audio)
                audio_net_est = audio_net_est.numpy(force=True)
                audio_net_est = np.squeeze(audio_net_est)

                mos = mos.numpy(force=True)
                mos = np.squeeze(mos)

                results.append(
                    [float(mos), float(est), int(dataset), float(audio_net_est)]
                )

            results = np.array(results)
            data_df = pd.DataFrame(
                results,
                columns=["MOS", "Estimation", "Dataset_Index", "AudioNet_Estimation"],
            )

            data_df["Dataset"] = dataset_split

            out_df = pd.concat([out_df, data_df])
            corr = np.corrcoef(results[:, 0], results[:, 1])[0, 1]
            print(f"{dataset_split} corr coef: {corr:6f}")

            metric_name = f"test_pearsons/{dataset_split}"
            metrics = dict()
            metrics[metric_name] = corr

            rmse = np.sqrt(np.mean((results[:, 0] - results[:, 1]) ** 2))
            metrics[f"test_rmse/{dataset_split}"] = rmse

            if dataset_split == "test":
                for dix in np.unique(data_df["Dataset_Index"]):
                    df_sub = data_df[data_df["Dataset_Index"] == dix]
                    corr = np.corrcoef(df_sub["MOS"], df_sub["Estimation"])[0, 1]
                    rmse = np.sqrt(np.mean((df_sub["MOS"] - df_sub["Estimation"]) ** 2))

                    corr_name = f"test_pearsons/dataset {dix}"
                    metrics[corr_name] = corr

                    rmse_name = f"test_rmse/dataset {dix}"
                    metrics[rmse_name] = rmse

            [logger.log_metrics(metrics) for logger in loggers]

    estimations_name = "estimations.csv"
    out_df.to_csv(estimations_name, index=False)

    # Store estimations to clearml
    if task is not None:
        task.upload_artifact(artifact_object=estimations_name, name="estimations csv")
        task.upload_artifact(artifact_object=out_df, name="estimations df")
    colormap = cm.rainbow
    # Plot estimations vs MOS
    for k, ds in enumerate(np.unique(out_df["Dataset"])):
        df_sub = out_df[out_df["Dataset"] == ds]
        mos = df_sub["MOS"]
        est = df_sub["Estimation"]
        dataset_index = df_sub["Dataset_Index"]
        plt.plot([1, 5], [1, 5], color="black", linestyle="dashed")
        plt.scatter(x=mos, y=est, c=dataset_index, alpha=0.1, cmap=colormap)
        corrcoef = np.corrcoef(mos, est)[0, 1]
        rmse = np.sqrt(np.mean((mos - est) ** 2))
        title_str = f"{ds} set, LCC={corrcoef:.4f}, RMSE={rmse:.4f}"
        for dx in np.unique(dataset_index):
            dx_ix = dataset_index == dx
            mos_dx = mos[dx_ix]
            est_dx = est[dx_ix]
            corrcoef_dx = np.corrcoef(mos_dx, est_dx)[0, 1]
            rmse_dx = np.sqrt(np.mean((mos_dx - est_dx) ** 2))
            subtitle_str = (
                f", (Dataset {dx}, LCC={corrcoef_dx:.4f}, RMSE={rmse_dx:.4f})"
            )
            title_str += subtitle_str
        plt.title(title_str)
        plt.xlabel("MOS")
        plt.ylabel("Estimation")
        plt.show()

    # audio_net vs aligner estimations
    for k, ds in enumerate(np.unique(out_df["Dataset"])):
        df_sub = out_df[out_df["Dataset"] == ds]
        mos = df_sub["AudioNet_Estimation"]
        est = df_sub["Estimation"]
        dataset_index = df_sub["Dataset_Index"]
        plt.plot([1, 5], [1, 5], color="black", linestyle="dashed")
        plt.scatter(x=mos, y=est, c=dataset_index, alpha=0.1, cmap=colormap)
        plt.title(f"{ds} set audio_net vs estimation ")
        plt.xlabel("audio_net estimation")
        plt.ylabel("final estimation")
        plt.show()

    for k, dx in enumerate(np.unique(out_df["Dataset_Index"])):
        xv = np.arange(0.5, 5.5, step=0.01)
        xv = torch.Tensor(xv)
        xv = xv[:, None]
        xv = xv.to(device)

        data_tensor = dx * torch.ones(xv.shape)
        data_tensor = data_tensor.squeeze()
        data_tensor = data_tensor.to(int)
        data_tensor = data_tensor.to(device)

        yv = model.network.aligner(xv, data_tensor)
        xv = xv.cpu().detach().numpy()
        yv = yv.cpu().detach().numpy()
        plt.plot([1, 5], [1, 5], color="black", linestyle="dashed")
        plt.scatter(xv, yv)
        plt.title(f"Alignment function for dataset {dx}")
        plt.xlabel("Raw score")
        plt.ylabel("Aligned")
        plt.show()


@hydra.main(config_path="alignnet/config", config_name="conf.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.logging.logger == "clearml":
        task = Task.init(
            project_name=cfg.project.name,
            task_name=cfg.project.task,
        )
    else:
        task = None
    print("Working directory : {}".format(os.getcwd()))

    # Seed
    seed = cfg.common.seed
    if seed is None:
        rng = np.random.default_rng()
        seed = rng.choice(10000)
        cfg.common.seed = seed
    pl.seed_everything(seed)

    # Transform
    transform = hydra.utils.instantiate(cfg.transform)

    data_class = hydra.utils.instantiate(cfg.dataclass)

    # TODO - replace with recursive instantiation, available in new Hydra
    audio_data = hydra.utils.instantiate(
        cfg.data, transform=transform, DataClass=data_class
    )

    num_datasets = len(cfg.data.data_dirs)

    # Lightning logs
    # Initialize tensorboard logger, letting hydra control directory and versions
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=".",
        name="",
        version="",
    )
    loggers = [tb_logger]
    [logger.log_hyperparams(dict(cfg)) for logger in loggers]

    checkpoint_callback = hydra.utils.instantiate(cfg.checkpoint)

    callbacks = [checkpoint_callback]
    if "earlystop" in cfg:
        # Earlystop needs monitor (e.g., val-loss) and mode (e.g., min). This can be added via CLI/cfg, otherwise steal the checkpoint values.
        stop_params = {"monitor": None, "mode": None}
        for k, _ in stop_params.items():
            if k in cfg.earlystop:
                stop_params[k] = cfg.earlystop[k]
            else:
                stop_params[k] = cfg.checkpoint[k]
        early_stopping_callback = hydra.utils.instantiate(cfg.earlystop, **stop_params)
        callbacks.append(early_stopping_callback)
    # Trainer
    trainer = hydra.utils.instantiate(
        cfg.optimization, callbacks=callbacks, logger=loggers
    )
    num_datasets = len(cfg.data.data_dirs)
    # Initialize network
    network = hydra.utils.instantiate(
        cfg.network, aligner={"num_datasets": num_datasets}
    )
    loss = hydra.utils.instantiate(cfg.loss)
    # TODO - replace with recursive instantiation, available in new Hydra
    optimizer = hydra.utils.instantiate(cfg.optimizer, lr=cfg.common.lr)

    # initialize model
    # TODO - update this with load_model function that uses trained_model folder
    if cfg.finetune.restore_file is not None:
        print(f"Loading model from checkpoint: {cfg.finetune.restore_file}")
        # initialize model
        model_class = hydra.utils.get_class(cfg.model._target_)
        # Path to pretrained model checkpoint
        model_path = os.path.join(cfg.finetune.restore_file, "model.ckpt")
        restore_cfg_path = os.path.join(cfg.finetune.restore_file, "config.yaml")

        with open(restore_cfg_path, "r") as f:
            restore_yaml = yaml.safe_load(f)
        restore_cfg = DictConfig(restore_yaml)

        restore_network = hydra.utils.instantiate(restore_cfg.network)
        # Turn restored audio_net gradients on or off depending on new network settings
        old_freeze_name = "audio_net_freeze_steps"
        if hasattr(network, old_freeze_name):
            frozen_steps = getattr(network, old_freeze_name)
        else:
            frozen_steps = network.audio_net_freeze_epochs
        if frozen_steps > 0:
            restore_network.set_audio_net_update_status(False)
        else:
            restore_network.set_audio_net_update_status(True)

        # Initialize identical network to pretrained version (necessary to appropriately load in aligner)
        # aligner is not transferable (different sizes based on number of datasets)

        restored_model = model_class.load_from_checkpoint(
            model_path, network=restore_network, loss=loss, optimizer=optimizer
        )
        # Grab audio_net from checkpoint
        network.audio_net = restored_model.network.audio_net

    model = hydra.utils.instantiate(
        cfg.model, network=network, loss=loss, optimizer=optimizer
    )
    print(model)
    # This is actually automatically stored in the .hydra folder...
    # Save a version of the config

    # Add working directory to config
    with open_dict(cfg):
        cfg.project.working_dir = os.getcwd()
    cfg_yaml = OmegaConf.to_yaml(cfg)
    cfg_out = "input_config.yaml"
    with open(cfg_out, "w") as file:
        file.write(cfg_yaml)

    if cfg.common.auto_batch_size:
        tuner = pl.tuner.Tuner(trainer)
        tuner.scale_batch_size(model, datamodule=audio_data)

    # Fit Trainer
    trainer.fit(model, audio_data)

    best_model_path = trainer.checkpoint_callback.best_model_path
    trained_model_path = "trained_model"
    os.makedirs(trained_model_path)

    # Save another copy of the top model
    top_model_path = os.path.join(trained_model_path, "model.ckpt")
    shutil.copy(best_model_path, top_model_path)
    print(f'experiment_path = "{os.getcwd()}"')
    print(f'model_ckpt = "{best_model_path}"')

    # Create output config
    output_config = DictConfig({})
    output_config.model = cfg.model
    output_config.network = cfg.network
    # Store num datasets directly
    output_config.network.aligner.num_datasets = num_datasets

    # Convert to yaml
    output_config = OmegaConf.to_yaml(output_config)

    # Save output config
    output_config_path = os.path.join(trained_model_path, "config.yaml")
    with open(output_config_path, "w") as file:
        file.write(output_config)

    # Get model class
    model_class = hydra.utils.get_class(cfg.model._target_)
    # Load best model
    model = model_class.load_from_checkpoint(best_model_path, network=network)
    post_train(model, audio_data, loggers, task=task)


if __name__ == "__main__":
    main()
