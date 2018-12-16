from nltk import word_tokenize
import nltk
import pandas as pd
from torchtext.data import Field, TabularDataset
from typing import List, Tuple
from collections import namedtuple
from sklearn.model_selection import train_test_split
import os
# nltk.download('punkt')

"""
Helper functions to load a toy NMT dataset from http://www.manythings.org/anki/
The .txt file for the given language pair should be in ./<src_lang>-<target_lang>.txt

Based on:
    https://towardsdatascience.com
        /how-to-use-torchtext-for-neural-machine-translation-plus-hack-to-make-it-5x-faster-77f3884d95
"""

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
Dataset = namedtuple('Dataset', ['src_field', 'target_field', 'train', 'val'])


def load_dataset(src: str, target: str) -> Tuple[List[str], List[str]]:
    file_path = './{}-{}.txt'.format(src, target)
    lines = open(file_path, encoding='utf-8').read().strip().split('\n')

    pairs = [[s for s in l.split('\t')] for l in lines]
    src_sentences = [p[0] for p in pairs]
    target_sentences = [p[1] for p in pairs]

    return src_sentences, target_sentences


def create_fields() -> Tuple[Field, Field]:
    def tokenize_src(sentence: str):
        return word_tokenize(sentence)

    def tokenize_target(sentence: str):
        return word_tokenize(sentence)

    src_field = Field(tokenize=tokenize_src, init_token='<sos>', eos_token='<eos>')
    target_field = Field(tokenize=tokenize_target, init_token='<sos>', eos_token='<eos>')

    return src_field, target_field


def convert_to_dataframe(src_sentences: List[str], target_sentences: List[str]) -> pd.DataFrame:
    raw_data = {
        'src': src_sentences,
        'target': target_sentences
    }
    df = pd.DataFrame(raw_data, columns=['src', 'target'])

    # remove very long sentences and sentences where len(source) is very different from len(translation)
    df['src_len'] = df['src'].str.count(' ')
    df['target_len'] = df['target'].str.count(' ')
    df = df.query('src_len < 80 & target_len < 80')
    df = df.query('target_len < src_len * 1.5 & target_len * 1.5 > src_len')

    return df


def save_train_test_dataset(src: str, target: str, test_size: float = 0.1) -> None:
    src_sentences, target_sentences = load_dataset(src, target)
    df = convert_to_dataframe(src_sentences, target_sentences)

    train, val = train_test_split(df, test_size=test_size)

    # save as csv
    train.to_csv('train_{}-{}.csv'.format(src, target), index=False)
    val.to_csv('val_{}-{}.csv'.format(src, target), index=False)


def create_dataset(src: str, target: str) -> Dataset:
    src_field, target_field = create_fields()
    data_fields = [(src, src_field), (target, target_field)]

    train_path = ROOT_DIR + '/train_{}-{}.csv'.format(src, target)
    val_path = ROOT_DIR + '/val_{}-{}.csv'.format(src, target)
    train, val = TabularDataset.splits(path='./', train=train_path, validation=val_path, format='csv',
                                       fields=data_fields)

    # index the tokens
    src_field.build_vocab(train, val)
    target_field.build_vocab(train, val)

    return Dataset(src_field, target_field, train, val)


if __name__ == "__main__":
    # only needs to be done once
    save_train_test_dataset('eng', 'deu')
