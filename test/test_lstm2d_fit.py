from unittest import TestCase
import torch
from model.lstm2d import LSTM2d


class LSTM2dFitTest(TestCase):
    """
    Unit tests to ensure the 2D-LSTM can learn something (fit a dataset).
    """
    embed_dim = 4
    encoder_state_dim = 4
    cell_state_dim = 5

    max_input_len = 3
    max_output_len = max_input_len

    vocab_size = 5

    def setUp(self):
        torch.manual_seed(42)
        device = torch.device('cpu')
        self.lstm = LSTM2d(embed_dim=self.embed_dim, state_dim_2d=self.cell_state_dim,
                           encoder_state_dim=self.encoder_state_dim, input_vocab_size=self.vocab_size,
                           output_vocab_size=self.vocab_size, device=device, dropout_p=0)

    def test_fits_small_dataset(self):
        """
        Tests if the model can fit a simple, small, random dataset (i.e. validate that it actually learns something).
        """
        dataset_size = 5
        x = torch.randint(1, self.vocab_size, (self.max_input_len, dataset_size), dtype=torch.long)
        x_lengths = torch.tensor([self.max_input_len]).repeat(dataset_size)
        y = x.clone()   # should learn the identity function

        loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.lstm.parameters(), lr=0.01)

        self.lstm.train()

        initial_loss = -1
        last_loss = -1
        for _ in range(200):
            y_pred = self.lstm.forward(x=x, y=y, x_lengths=x_lengths).view(-1, self.vocab_size)
            loss_value = loss(y_pred, y.view(-1))

            last_loss = loss_value.item()
            if initial_loss < 0:
                initial_loss = last_loss

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        # print("from {} to {}".format(initial_loss, last_loss))
        self.assertTrue(last_loss < 0.1 * initial_loss, 'The model did not learn enough.')
