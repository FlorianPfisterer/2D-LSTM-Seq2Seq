import nltk
import pandas as pd
from torchtext.data import Field, TabularDataset
from typing import List, Tuple
from collections import namedtuple
import os

nltk.download('punkt')

"""
Helper functions to preprocess / load the small iwslt14 eng-deu NMT dataset from
    https://github.com/harvardnlp/var-attn/tree/master/data.
Based on:
    https://towardsdatascience.com
        /how-to-use-torchtext-for-neural-machine-translation-plus-hack-to-make-it-5x-faster-77f3884d95
"""

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
Dataset = namedtuple('Dataset', ['src', 'tgt', 'train', 'val'])

BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'


def __load_dataset(mode: str = 'train') -> Tuple[List[str], List[str]]:
    """
    Reads the dataset's .txt files for the given mode.
    Args:
        mode: the mode in which to read the dataset, must be one of ['train', 'val']

    Returns:
        a tuple of two lists:
            src: list of sentences in the source language
            tgt: list of sentences in the target language
    """
    assert mode in ['train', 'val']

    src_file_path = os.path.join(ROOT_DIR, 'src-{}.txt'.format(mode))
    src = open(src_file_path, encoding='utf-8').read().split('\n')

    tgt_file_path = os.path.join(ROOT_DIR, 'tgt-{}.txt'.format(mode))
    tgt = open(tgt_file_path, encoding='utf-8').read().split('\n')

    return src, tgt


def __create_fields() -> Tuple[Field, Field]:
    """
    Creates torchtext-Fields for source and target language

    Returns:
        a tuple of two torchtext.Field s:
            - src_field: Field representing the source language
            - tgt_field: Field representing the target language
    """
    src_field = Field(include_lengths=True, pad_token=PAD_TOKEN)
    tgt_field = Field(include_lengths=True, init_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN)

    return src_field, tgt_field


def __convert_to_df(src: List[str], tgt: List[str]) -> pd.DataFrame:
    """
    Converts the given two lists of source and target sentences into one pandas.DataFrame.
    Args:
        src: the list of source sentences
        tgt: the list of target sentences, must be of equal length as src

    Returns:
        a pandas.DataFrame object with two columns, 'src' and 'tgt' containing the given lists
    """
    assert len(src) == len(tgt)

    raw_data = {
        'src': src,
        'tgt': tgt
    }
    df = pd.DataFrame(raw_data, columns=['src', 'tgt'])

    return df


def __save_dataset_to_csv_if_needed(mode: str) -> None:
    """
    Saves the dataset with the given mode to disk in a <mode>.csv file -- if this file does not already exist.
    Args:
        mode: the mode in which to read the dataset, must be one of ['train', 'val']
    """
    file_name = '{}.csv'.format(mode)
    file_path = os.path.join(ROOT_DIR, file_name)
    if not os.path.exists(file_path):
        src, tgt = __load_dataset(mode)
        frame = __convert_to_df(src, tgt)
        frame.to_csv(file_path, index=False, header=False)


def save_dataset() -> None:
    __save_dataset_to_csv_if_needed('train')
    __save_dataset_to_csv_if_needed('val')


def create_dataset() -> Dataset:
    """
    Creates a Dataset tuple that allows access to vocabularies in src and tgt language as well as the training and
    validation data.

    Returns:
        a Dataset tuple with the following values:
            - src: torchtext.Field for the source language
            - tgt: torchtext.Field for the target language
            - train: torchtext.Dataset wrapping the training data
            - val: torchtext.Dataset wrapping the validation data
    """
    save_dataset()

    src, tgt = __create_fields()
    data_fields = [('src', src), ('tgt', tgt)]

    train, val = TabularDataset.splits(path=ROOT_DIR, train='train.csv', validation='val.csv', format='csv',
                                       fields=data_fields)

    # index the tokens
    src.build_vocab(train, val)
    tgt.build_vocab(train, val)

    return Dataset(src, tgt, train, val)


if __name__ == "__main__":
    save_dataset()
