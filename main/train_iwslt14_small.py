from data.iwslt14_small.dataset_utils import create_dataset, BOS_TOKEN, EOS_TOKEN
from model.lstm2d import LSTM2d
from torchtext.data import BucketIterator
import argparse
import torch
import numpy as np

# define options
parser = argparse.ArgumentParser(description='train_iwslt14_small.py')
parser.add_argument('-batch_size', default=32,
                    help='The batch size to use for training and inference.')
parser.add_argument('-epochs', default=1,
                    help='The number of epochs to train.')
parser.add_argument('-shuffle', default=True,
                    help='Whether or not to shuffle the training examples.')
parser.add_argument('-lr', default=1e-3,
                    help='The learning rate to use.')
parser.add_argument('-embed_dim', default=100,
                    help='The dimension of the embedding vectors for both the source and target language.')
parser.add_argument('-encoder_state_dim', default=64,
                    help='The dimension of the bidirectional encoder LSTM states.')
parser.add_argument('-state_2d_dim', default=128,
                    help='The dimension of the 2D-LSTM hidden & cell states.')
options = parser.parse_args()


def main():
    dataset = create_dataset()

    src_vocab_size = len(dataset.src.vocab)
    tgt_vocab_size = len(dataset.tgt.vocab)
    bos_token = dataset.tgt.vocab.stoi[BOS_TOKEN]
    eos_token = dataset.tgt.vocab.stoi[EOS_TOKEN]

    model = LSTM2d(
        embed_dim=options.embed_dim,
        state_dim_2d=options.state_2d_dim,
        encoder_state_dim=options.encoder_state_dim,
        input_vocab_size=src_vocab_size,
        output_vocab_size=tgt_vocab_size,
        bos_token=bos_token,
        eos_token=eos_token
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr)
    loss = torch.nn.CrossEntropyLoss()

    train_iter = BucketIterator(dataset.train, batch_size=options.batch_size,
                                sort_key=lambda x: -len(x.src), shuffle=options.shuffle)

    for epoch in range(options.epochs):
        print('Starting epoch #{}'.format(epoch + 1))

        loss_history = []
        model.train()

        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()
            x = batch.src
            y = batch.tgt[1:, :]        # remove <sos> token (the net should not generate this)

            y_pred = model.forward(x=x, y=y)
            y_pred = y_pred.view(-1, tgt_vocab_size)

            loss_value = loss(y_pred, y.view(-1))
            loss_history.append(loss_value.item())

            loss_value.backward()
            optimizer.step()

            if i > 0 and not i % 10:
                avg_loss = np.mean(loss_history)
                print('Average loss after {} batches (epoch #{}): {}'.format(i, epoch + 1, avg_loss))


if __name__ == '__main__':
    main()
