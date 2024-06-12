import hydra
import os
import torch
import warnings

import numpy as np
import pandas as pd

from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

import alignnet

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


@hydra.main(
    config_path="./inference_configs", config_name="config.yaml", version_base=None
)
def main(cfg: DictConfig) -> None:
    """
    Run inference on data with a trained model.

    See `python inference.py --help` for more details.
    """

    # Transform
    transform = hydra.utils.instantiate(cfg.transform)

    print("Initializing data")
    audio_data = hydra.utils.instantiate(
        cfg.data,
        transform=transform,
    )
    print(f"Loading model from {cfg.model.path}")
    model = alignnet.load_model(cfg.model.path)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if cfg.model.dataset_index == "reference":
        dataset_index = torch.tensor([model.network.aligner.reference_index])
    else:
        dataset_index = torch.tensor([cfg.model.dataset_index])
    dataset_index = dataset_index.to(device)
    # Switch to eval mode
    model.eval()
    with torch.no_grad():
        output_dicts = []
        print(f"Generating estimations")
        for ix, (audio, mos, dataset) in enumerate(
            tqdm(audio_data, total=len(audio_data))
        ):
            # Make audio look batched
            audio = audio[None, None, :]
            audio = audio.to(device)

            est = model(audio, dataset_index)
            audio_path = audio_data.score_file.loc[ix, audio_data.pathcol]
            output_dicts.append(
                {
                    "file": audio_path,
                    "estimate": est.to("cpu").numpy()[0],
                    "dataset": dataset,
                    "AlignNet dataset index": dataset_index.to("cpu").numpy()[0]
                }
            )
            # Iterating over Datasets does not always stop appropriately so this ensures it does
            if ix == len(audio_data) - 1:
                break
    output_df = pd.DataFrame(output_dicts)
    print("First 5 results:")
    print(output_df.head())
    output_dir = os.path.dirname(cfg.output.file)

    if not os.path.exists(output_dir) and output_dir != "":
        os.makedirs(output_dir)
    print(f"Saving results to {cfg.output.file}")
    output_df.to_csv(cfg.output.file, index=False)


if __name__ == "__main__":
    main()
