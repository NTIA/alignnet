import os

import numpy as np
import pandas as pd

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

EXT = ".csv"


def get_split_numbers(n_audio, split_fraction):
    """
    Number of audio files per split

    Parameters
    ----------
    n_audio : int
        Number of audio files
    split_fraction : list
        List of fractions for each split

    Returns
    -------
    list
        List of number of items in each split.
    """
    # Number of audio files per split
    split_numbers = []
    for split_frac in split_fraction[:-1]:
        split_num = np.round(split_frac * n_audio)
        split_numbers.append(int(split_num))
    split_numbers.append(n_audio - np.sum(split_numbers))
    return split_numbers


def split_df_by_column(df, split_col, split_names, split_fraction):
    """
    Generate dictionary of indices for splitting up a DataFrame while maintaining
    balance within splits for a single column.

    Split dataframe while maintaining balance of elements within a specific column.

    Note that if there are n conditions labelled within a certain column, this  
    ensures that the proper ratio of conditions is maintained within the train, validation, 
    and test datasets. For example, if 80% of the data is condition A, 
    15% is condition B, and 5% is condition C, then those percentage ratios will 
    be preserved in each of the train, validation, and test datasets.


    Parameters
    ----------
    df : pd.DataFrame
        _description_
    split_col : str
        Categorical column name in df that will have balance of values preserved
        in each output dataset.
    split_names : list
        List of names for output csvs, used as keys in output dict.
    split_fraction : list
        Fraction of df to go into each dictionary item (ordered according to split_names).

    Returns
    -------
    dict
        Dictionary with keys being split_names and values being array of indices
        that has length of len(df) * split_fraction for each element.
    """
    column_vals = np.unique(df[split_col])

    # Initialize empty dictionary
    split_ix = dict()
    for name in split_names:
        split_ix[name] = []

    for col_val in column_vals:
        df_filt = df[df[split_col] == col_val]
        split_ix_val = split_df(df_filt, split_names, split_fraction)
        for name, ix in split_ix_val.items():
            split_ix[name].extend(ix)

    # One final shuffle
    rng = np.random.default_rng()
    for name, ix in split_ix.items():
        split_ix[name] = rng.choice(ix, len(ix), replace=False)
    return split_ix


def split_df(df, split_names, split_fraction):
    """
    Generate dictionary of indices for splitting up a DataFrame.

    Dictionary keys are defined by split_names and the number of items in each key
    is determined by split_fraction.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to split.
    split_names : list
        List of names for output csvs, used as keys in output dict.
    split_fraction : list
        Fraction of df to go into each dictionary item (ordered according to split_names).

    Returns
    -------
    dict
        Dictionary with keys being split_names and values being array of indices
        that has length of len(df) * split_fraction for each element.
    """
    # Number of rows in df
    n_audio = len(df)

    # Get list with number of audio files per split
    split_numbers = get_split_numbers(n_audio, split_fraction)

    # Initialize random number generator
    rng = np.random.default_rng()

    # Shuffle index
    shuffled_ix = rng.choice(df.index, size=len(df.index), replace=False)
    split_ix = dict()
    seen = 0
    for n, name in zip(split_numbers, split_names):
        start = seen
        end = start + n
        split_ix[name] = shuffled_ix[start:end]
        seen = end
    return split_ix


def main(args, n=None):
    if len(args.split_fraction) != len(args.split_names):
        raise ValueError(
            f"Split fraction and split names must be same length, {len(args.split_fraction)} != {len(args.split_names)}"
        )
    output_dir = args.output_dir

    if n is not None:
        output_dir += f"/split{n:02}"
    os.makedirs(output_dir, exist_ok=True)
    # Read scores
    score_df = pd.read_csv(args.label_file)

    if args.split_column is None:
        # Split all the files
        split_ix = split_df(score_df, args.split_names, args.split_fraction)
    else:
        split_ix = split_df_by_column(
            # Split according to the split_column
            score_df,
            args.split_column,
            args.split_names,
            args.split_fraction,
        )

    ext = ".csv"
    for name, ix in split_ix.items():
        audio = score_df.iloc[ix]
        out_name = os.path.join(output_dir, name + ext)
        audio.to_csv(out_name, index=False)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Split a label_file containing target and pathcol for audio file into train, valid, and test csvs.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "label_file",
        type=str,
        help=("Path and filename to file with subjective scores and file paths."),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data-splits",
        help="Path where data splits will be stored.",
    )

    parser.add_argument(
        "--split-names",
        nargs="+",
        default=["train", "valid", "test"],
        help="Labels for how data is split and saved.",
    )

    parser.add_argument(
        "--split-fraction",
        nargs="+",
        type=float,
        default=[0.8, 0.1, 0.1],
        help="Amount of data to use for each split-name. Must sum to 1 and be same length as --split-names.",
    )

    parser.add_argument(
        "--split-column",
        type=str,
        default=None,
        help=(
            "Column for which data should be split according to split-fraction (e.g., force distributions of values in "
            "that column across each dataset.)"
        ),
    )

    parser.add_argument(
        "--n-splits", type=int, default=1, help="Number of independent splits to make."
    )

    parser.add_argument(
        "--no-header", action="store_true", help="Flag for no header in csvs."
    )

    args = parser.parse_args()
    for k in range(args.n_splits):
        main(args, n=k)
