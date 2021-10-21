from typing import Optional
from pathlib import Path
import numpy as np


def load_dataset(dataset: str, small: bool = False):
    # Trick to get homeworks
    # First check current directory
    data_path: Optional[Path] = None
    if (Path(".") / "data").exists():
        data_path = Path(".") / "data"
    else:
        cur_dir_parents = Path(".").absolute().parents
        for parent in cur_dir_parents:
            if (parent / "data").exists():
                data_path = parent / "data"
                break

    if data_path is None:
        print("Could not find dataset. Please run from within 446 hw folder.")
        exit(0)

    if dataset.lower() == "mnist":
        with np.load(data_path / "mnist" / "mnist.npz", allow_pickle=True) as f:
            X_train, labels_train = f['x_train'], f['y_train']
            X_test, labels_test = f['x_test'], f['y_test']

        # Reshape each image of size 28 x 28 to a vector of size 784
        X_train = X_train.reshape(-1, 784)
        X_test = X_test.reshape(-1, 784)

        # Pixel values are integers from 0 to 255.
        # Dividing the pixel values by 255 is a technique called normalization,
        # which is a standard practice in Machine Learning to prevent large numeric values
        X_train = X_train / 255
        X_test = X_test / 255

        return ((X_train, labels_train), (X_test, labels_test))

    if dataset.lower() == "polyreg":
        f = open((data_path / "polyreg" / "polydata.dat"), "r")
        allData = np.loadtxt(f, delimiter=",")
        f.close()
        return allData

    if dataset.lower() == "crime":
        import pandas as pd
        df_train = pd.read_table(data_path / "crime-data" / "crime-train.txt")
        df_test = pd.read_table(data_path / "crime-data" / "crime-test.txt")
        return df_train, df_test

    if dataset.lower() == "xor":
        return (
            (np.load(data_path / "xor" / "x_train.npy"), np.load(data_path / "xor" / "y_train.npy")),
            (np.load(data_path / "xor" / "x_val.npy"), np.load(data_path / "xor" / "y_val.npy")),
            (np.load(data_path / "xor" / "x_test.npy"), np.load(data_path / "xor" / "y_test.npy")),
        )
