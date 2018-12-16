from data.toy_nmt_dataset import create_dataset
from torchtext.data import BucketIterator
from model.lstm2d import LSTM2d
import torch
import numpy as np

BATCH_SIZE = 128
EPOCHS = 1

EMBED_DIM = 100
ENCODER_STATE_DIM = 64
STATE_DIM_2D = 128


def main():
    dataset = create_dataset('eng', 'deu')
    train_iter = BucketIterator(dataset.train, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.target), shuffle=True)

    src_vocab_size = len(dataset.src_field.vocab)           # 16,449
    target_vocab_size = len(dataset.target_field.vocab)     # 32,997

    model = LSTM2d(
        embed_dim=EMBED_DIM,
        state_dim_2d=STATE_DIM_2D,
        encoder_state_dim=ENCODER_STATE_DIM,
        input_vocab_size=src_vocab_size,
        output_vocab_size=target_vocab_size
    )

    optimizer = torch.optim.Adam(model.parameters())
    loss = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        print('Starting training epoch #{}'.format(epoch + 1))

        model.train()
        for i, batch in enumerate(train_iter):
            x = batch.eng
            y = batch.deu
            y_t = y.t()

            y_pred = model.forward(x, y).permute(1, 2, 0)
            loss_value = loss(y_pred, y_t)
            total_loss += loss_value.item()

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            if i > 0 and not i % 10:
                print('Loss in batch #{}: {}'.format(i, loss_value))

        print('Loss after epoch #{}: {}'.format(epoch + 1, total_loss))
        run_validation(model, dataset.val)


def run_validation(model, val):
    val_iter = BucketIterator(val, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.target), shuffle=True)

    model.eval()
    accuracies, losses = [], []
    loss = torch.nn.CrossEntropyLoss()
    for batch in iter(val_iter):
        x = batch.eng
        y = batch.deu
        y_t = y.t()

        with torch.no_grad():
            y_pred = model.forward(x).permute(1, 2, 0)  # (batch x output_vocab_size x sequence_len)
            loss_value = loss(y_pred, y_t).item()
            losses.append(loss_value)

            num_correct = torch.argmax(y_pred, dim=1).eq(y_t).sum()
            accuracies.append(num_correct / BATCH_SIZE)

    mean_acc = np.mean(np.array(accuracies))
    mean_loss = np.mean(np.array(losses))
    print('Validation metrics: mean accuracy = {}, mean loss = {}'.format(mean_acc, mean_loss))


if __name__ == '__main__':
    main()
