import datetime
import os
import pickle
import torchaudio

import pandas as pd

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch import save
from tqdm import tqdm

from alignnet import transforms


def load_audio(fpath, target_fs=None):
    """
    Load audio file and resample as necessary.

    Parameters
    ----------
    fpath : str
        Path to audio file.
    target_fs : int, optional
        Target sample rate if resampling required, by default None

    Returns
    -------
    torch.tensor
        Audio file
    int
        Sample rate
    """
    audio, sample_rate = torchaudio.load(fpath)
    if target_fs is not None and sample_rate != target_fs:
        # We have a target fs and the current sample rate is not it => resample!
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_fs,
            dtype=audio.dtype,
        )
        audio = resampler(audio)
        sample_rate = target_fs
    return audio, sample_rate


def transform_path_walk(datapath, outpath, transform, target_fs):
    """
    Transform all audio files within directory via os.walk

    Parameters
    ----------
    datapath : str
        Path to walk
    outpath : str
        Path to save transformed files
    transform : transform
        Transform from alignnet.transforms. Must have transform.transform(audio, **kwargs)
        as a method.
    target_fs : int
        Target sample rate. Audio will be resampled to this prior to transform if needed.
    """
    failed_files = []
    for path, directories, files in os.walk(datapath):
        print(f"Transforming audio in: {path}")
        for file in tqdm(files):
            bname, ext = os.path.splitext(file)
            if ext == ".wav":
                fpath = os.path.join(path, file)
                try:
                    audio, sample_rate = load_audio(fpath, target_fs=target_fs)
                except:
                    failed_files.append(fpath)
                    continue
                audio = transform.transform(audio, sample_rate=sample_rate)
                audio = audio.float()

                # Get relative path from datapath to path
                subpath = os.path.relpath(path, datapath)
                # Make a new path from outpath to path
                newpath = os.path.join(outpath, subpath)

                # Make directories if necessary
                if not os.path.exists(newpath):
                    os.makedirs(newpath)

                outfile = os.path.join(newpath, bname + ".pkl")
                outfile = os.path.abspath(outfile)

                with open(outfile, "wb") as output:
                    pickle.dump(audio, output)
    print(f"Unable to transform following files:\n{failed_files}")


def transform_csv(
    datapath, outpath, csv_list, transform, target_fs, pathcol="filename"
):
    """
    Transform all audio files listed in csv.

    Parameters
    ----------
    datapath : str
        Parent path to (potential) relative path within csv pathcol.
    outpath : str
        Path where transformed audio will be saved.
    csv_list : str
        Path to csv file containing audio names to transform in pathcol
    transform : transform
        Transform from alignnet.transforms. Must have transform.transform(audio, **kwargs)
        as a method.
    target_fs : int
        Target sample rate. Audio will be resampled to this prior to transform if needed.
    pathcol : str, optional
        Column in csv that contains audio file names, by default "filename"
    """
    # Load csv int dataframe
    df = pd.read_csv(csv_list)
    for ix, row in tqdm(df.iterrows(), total=len(df)):
        # Get file name
        fname = row[pathcol]
        # Create file path
        fpath = os.path.join(datapath, fname)

        # Load audio and transform it
        audio, sample_rate = load_audio(fpath, target_fs=target_fs)
        audio = transform.transform(audio, sample_rate=sample_rate)
        audio = audio.float()

        outfile = os.path.join(outpath, fname)
        newpath = os.path.dirname(outfile)

        # Make directories if necessary
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        outfile, _ = os.path.splitext(outfile)
        outfile = outfile + ".pkl"

        with open(outfile, "wb") as output:
            pickle.dump(audio, output)


def main(datapath, outpath, transform_name, csv_list, **kwargs):
    log_file = os.path.join(outpath, "readme.log")
    if transform_name == "Mel":
        transform = transforms.MelTransform()
    elif transform_name == "STFT":
        transform = transforms.STFTTransform()
    for k, v in kwargs.items():
        if hasattr(transform, k):
            setattr(transform, k, v)
    # We never want to flatten
    if hasattr(transform, "flatten"):
        setattr(transform, "flatten", False)
    os.makedirs(outpath, exist_ok=True)
    with open(log_file, "w") as outfile:
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        time_str = f"Running {__file__}\nStarted: {time}\n"
        outfile.write(time_str)
        input_str = f"datapath={datapath}, outpath={outpath}, transform_name={transform_name}, csv_list={csv_list}\nkwargs: "
        for k, v in kwargs.items():
            input_str += f"{k}={v}, "
        outfile.write(f"Inputs: {input_str}")
        transform_str = f"type: {type(transform)}\nAttributes: "
        for v in dir(transform):
            if v[0] != "_":
                transform_str += f"{v}={getattr(transform, v)}, "
        outfile.write(transform_str)
    c = 0

    _, ext = os.path.splitext(datapath)
    if csv_list is not None:
        print("Load csv and transform files in there")
        transform_csv(
            datapath=datapath,
            outpath=outpath,
            csv_list=csv_list,
            transform=transform,
            target_fs=kwargs["target_fs"],
        )
    else:
        print("Walk through the input directory")
        transform_path_walk(
            datapath=datapath,
            outpath=outpath,
            transform=transform,
            target_fs=kwargs["target_fs"],
        )

    with open(log_file, "a") as outfile:
        finish_str = (
            f'\nFinished: {datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}\n'
        )
        outfile.write(finish_str)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "transform_name",
        choices=["Mel", "STFT"],
        help="Transform to apply. Corresponds to a transform class in alignnet.transforms",
    )
    parser.add_argument("datapath", type=str, help="Path to data to transform")
    parser.add_argument(
        "outpath",
        type=str,
        help=("Path where transformed version of data is stored."),
    )

    parser.add_argument(
        "--target-fs",
        default=None,
        type=int,
        help="Sample rate to resample audio to prior to transformation. If None, no resampling done.",
    )

    parser.add_argument(
        "--fft-win-length", default=512, type=int, help="Window length for an STFT."
    )

    parser.add_argument(
        "--win-overlap", default=256, type=int, help="Window overlap for an STFT."
    )

    parser.add_argument(
        "--csv-list",
        type=str,
        default=None,
        help=(
            "CSV file with list of files to transform. Assumes that "
            "os.path.join(datapath, x) is the full path to a file, where x is a "
            "row of the csv under column 'filename'"
        ),
    )

    parser.add_argument(
        "--log",
        action="store_true",
        help="Take log10 of representations.",
    )


    args = parser.parse_args()

    main(**vars(args))
