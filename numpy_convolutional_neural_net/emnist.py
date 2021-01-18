"""
Slightly modified version of https://github.com/hsjeong5/MNIST-for-Numpy
"""

import gzip
import os
import pickle
from urllib import request

import numpy as np

filename = [
    ["training_images", "emnist-balanced-train-images-idx3-ubyte.gz"],
    ["test_images", "emnist-balanced-test-images-idx3-ubyte.gz"],
    ["training_labels", "emnist-balanced-train-labels-idx1-ubyte.gz"],
    ["test_labels", "emnist-balanced-test-labels-idx1-ubyte.gz"]
]


def download_emnist():
    base_url = "http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/"
    name = "gzip.zip"
    
    print("Downloading " + name + "...")
    request.urlretrieve(base_url + name, name)
    print("Download complete.")

    with gzip.open(name, 'rb') as z:
        for name in filename:
            with z.open(name[1], 'rb') as f:
                f.extractall()



def save_mnist():
    emnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            emnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            emnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)

    for _, gz_file in filename:
        os.remove(gz_file)

    with open("emnist.pkl", 'wb') as f:
        pickle.dump(emnist, f)
    print("Save complete.")


def init():
    if not os.path.isfile("emnist.pkl"):
        download_emnist()
        save_mnist()
    else:
        print("Dataset already downloaded, delete emnist.pkl if you want to re-download.")


def load():
    with open("emnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

init()