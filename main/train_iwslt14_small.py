from data.iwslt14_small.dataset_utils import create_dataset, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from model.lstm2d import LSTM2d
from data.sorted_batch_iterator import SortedBatchIterator
import argparse
import torch
import numpy as np

# define options
parser = argparse.ArgumentParser(description='train_iwslt14_small.py')
parser.add_argument('-batch_size', default=3,
                    help='The batch size to use for training and inference.')
parser.add_argument('-epochs', default=5,
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
    pad_token = dataset.tgt.vocab.stoi[PAD_TOKEN]

    model = LSTM2d(
        embed_dim=options.embed_dim,
        state_dim_2d=options.state_2d_dim,
        encoder_state_dim=options.encoder_state_dim,
        input_vocab_size=src_vocab_size,
        output_vocab_size=tgt_vocab_size,
        bos_token=bos_token,
        eos_token=eos_token,
        pad_token=pad_token
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr)
    train_iter = SortedBatchIterator(dataset.train, sort_key=lambda example: (-len(example.src), -len(example.tgt)),
                                     batch_size=options.batch_size, shuffle=options.shuffle)

    for epoch in range(options.epochs):
        print('Starting epoch #{}'.format(epoch + 1))

        loss_history = []
        model.train()

        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()
            x, x_lengths = batch.src
            y = batch.tgt
            y = y[1:, :]                # remove <sos> token (the net should not generate this)

            x_lengths[x_lengths <= 0] = 1   # TODO -- crashes for values <= 0

            y_pred = model.forward(x=x, x_lengths=x_lengths, y=y)
            loss_value = model.loss(y_pred, y)
            loss_history.append(loss_value.item())

            loss_value.backward()
            optimizer.step()

            if i > 0 and not i % 10:
                avg_loss = np.mean(loss_history)
                print('Average loss after {} batches (epoch #{}): {}'.format(i, epoch + 1, avg_loss))

            if not i % 100:
                model.eval()
                test_model(model, dataset)
                model.train()


def test_model(model, dataset):
    example_sentence = 'Good morning , how are you ?'
    tokens = example_sentence.split(' ')
    x = torch.tensor([[dataset.src.vocab.stoi[w] for w in tokens]], dtype=torch.long).t()
    x_lengths = torch.tensor([len(tokens)], dtype=torch.long)
    pred = model.forward(x, x_lengths)

    predicted_tokens = list(torch.argmax(pred, dim=-1).view(-1))
    output_sentence = ' '.join([dataset.tgt.vocab.itos[i] for i in predicted_tokens])
    print('translate(\"{}\") ==> \"{}\"'.format(example_sentence, output_sentence))


if __name__ == '__main__':
    main()
