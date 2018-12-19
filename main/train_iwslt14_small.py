from data.iwslt14_small.dataset_utils import create_dataset, BOS_TOKEN, EOS_TOKEN
from torchtext.data import BucketIterator
import numpy as np


def main():
    dataset = create_dataset()
    train_iter = BucketIterator(dataset.train, batch_size=5, sort_key=lambda x: len(x.src), shuffle=False)

    batch = next(iter(train_iter))

    src = batch.src.t().numpy().tolist()
    src_vocab = dataset.src.vocab.itos
    for input in src:
        tokens = [src_vocab[idx] for idx in input]
        print(' '.join(tokens).replace('<pad>', ''))

    tgt = batch.tgt.t().numpy().tolist()
    tgt_vocab = dataset.tgt.vocab.itos
    for output in tgt:
        tokens = [tgt_vocab[idx] for idx in output]
        print(' '.join(tokens).replace('<pad>', ''))

    bos_token = dataset.tgt.vocab.stoi[BOS_TOKEN]
    eos_token = dataset.tgt.vocab.stoi[EOS_TOKEN]

    print(bos_token)
    print(eos_token)


if __name__ == '__main__':
    main()
