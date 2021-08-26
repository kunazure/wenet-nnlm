#!/usr/bin/env python3

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This code is ported from the following implementation written in Torch.
# https://github.com/chainer/chainer/blob/master/examples/ptb/train_ptb_custom_loop.py

import h5py
import logging
import numpy as np
import os
import random
import six
from tqdm import tqdm


def load_dataset(path, label_dict, outdir=None):
    """Load and save HDF5 that contains a dataset and stats for LM

    Args:
        path (str): The path of an input text dataset file
        label_dict (dict[str, int]):
            dictionary that maps token label string to its ID number
        outdir (str): The path of an output dir

    Returns:
        tuple[list[np.ndarray], int, int]: Tuple of
            token IDs in np.int32 converted by `read_tokens`
            the number of tokens by `count_tokens`,
            and the number of OOVs by `count_tokens`
    """
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        filename = outdir + "/" + os.path.basename(path) + ".h5"
        if os.path.exists(filename):
            logging.info(f"loading binary dataset: {filename}")
            f = h5py.File(filename, "r")
            return f["data"][:], f["n_tokens"][()], f["n_oovs"][()]
    else:
        logging.info("skip dump/load HDF5 because the output dir is not specified")
    logging.info(f"reading text dataset: {path}")
    ret = read_tokens(path, label_dict)
    n_tokens, n_oovs = count_tokens(ret, label_dict["<unk>"])
    if outdir is not None:
        logging.info(f"saving binary dataset: {filename}")
        with h5py.File(filename, "w") as f:
            # http://docs.h5py.org/en/stable/special.html#arbitrary-vlen-data
            data = f.create_dataset(
                "data", (len(ret),), dtype=h5py.special_dtype(vlen=np.int32)
            )
            data[:] = ret
            f["n_tokens"] = n_tokens
            f["n_oovs"] = n_oovs
    return ret, n_tokens, n_oovs


def read_tokens(filename, label_dict):
    """Read tokens as a sequence of sentences

    :param str filename : The name of the input file
    :param dict label_dict : dictionary that maps token label string to its ID number
    :return list of ID sequences
    :rtype list
    """

    data = []
    unk = label_dict["<unk>"]
    for ln in tqdm(open(filename, "r", encoding="utf-8")):
        data.append(
            np.array(
                [label_dict.get(label, unk) for label in ln.split()], dtype=np.int32
            )
        )
    return data


def count_tokens(data, unk_id=None):
    """Count tokens and oovs in token ID sequences.

    Args:
        data (list[np.ndarray]): list of token ID sequences
        unk_id (int): ID of unknown token

    Returns:
        tuple: tuple of number of token occurrences and number of oov tokens

    """

    n_tokens = 0
    n_oovs = 0
    for sentence in data:
        n_tokens += len(sentence)
        if unk_id is not None:
            n_oovs += np.count_nonzero(sentence == unk_id)
    return n_tokens, n_oovs


def compute_perplexity(result):
    """Computes and add the perplexity to the LogReport

    :param dict result: The current observations
    """
    # Routine to rewrite the result dictionary of LogReport to add perplexity values
    result["perplexity"] = np.exp(result["main/loss"] / result["main/count"])
    if "validation/main/loss" in result:
        result["val_perplexity"] = np.exp(result["validation/main/loss"])



def make_lexical_tree(word_dict, subword_dict, word_unk):
    """Make a lexical tree to compute word-level probabilities"""
    # node [dict(subword_id -> node), word_id, word_set[start-1, end]]
    root = [{}, -1, None]
    for w, wid in word_dict.items():
        if wid > 0 and wid != word_unk:  # skip <blank> and <unk>
            if True in [c not in subword_dict for c in w]:  # skip unknown subword
                continue
            succ = root[0]  # get successors from root node
            for i, c in enumerate(w):
                cid = subword_dict[c]
                if cid not in succ:  # if next node does not exist, make a new node
                    succ[cid] = [{}, -1, (wid - 1, wid)]
                else:
                    prev = succ[cid][2]
                    succ[cid][2] = (min(prev[0], wid - 1), max(prev[1], wid))
                if i == len(w) - 1:  # if word end, set word id
                    succ[cid][1] = wid
                succ = succ[cid][0]  # move to the child successors
    return root
