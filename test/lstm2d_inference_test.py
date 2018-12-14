from unittest import TestCase
import torch
from model.lstm2d import LSTM2d


class LSTM2dInferenceTest(TestCase):
    """
    Unit tests for the 2D-LSTM in inference mode.
    """
    embed_dim = 50
    encoder_state_dim = 20
    cell_state_dim = 25

    batch_size = 42

    max_input_len = 4
    max_output_len = 5

    vocab_size = 3

    def setUp(self):
        torch.manual_seed(42)

        self.lstm = LSTM2d(embed_dim=self.embed_dim, state_dim_2d=self.cell_state_dim,
                           encoder_state_dim=self.encoder_state_dim, max_input_len=self.max_input_len,
                           max_output_len=self.max_output_len, vocab_size=self.vocab_size)

    def test_dimensions(self):
        """
        Tests if the input and output dimensions of the 2D-LSTM are as expected.
        """
        # random token indices of shape (max_input_len x batch_size)
        sample_x = torch.randint(0, self.vocab_size, (self.max_input_len, self.batch_size), dtype=torch.long)

        # toy inference
        self.lstm.eval()
        pred = self.lstm.forward(x=sample_x, y=None)

        pred_shape = list(pred.shape)
        self.assertEqual(pred_shape, [self.max_output_len, self.batch_size, self.vocab_size],
                         'The predictions have an unexpected shape.')

    def test_valid_softmax(self):
        """
        Tests if the output predictions of the 2D-LSTM form a valid softmax distribution over the vocabulary, i.e.
        the elements are in [0, 1] and sum to 1.
        """
        sample_x = torch.randint(0, self.vocab_size, (self.max_input_len, self.batch_size), dtype=torch.long)

        self.lstm.eval()
        pred = self.lstm.forward(x=sample_x, y=None)    # shape (max_output_len x batch_size x vocab_size)

        # check [0, 1] range
        self.assertTrue(torch.max(pred) <= 1.0, 'Softmax distribution contains values > 1.')
        self.assertTrue(torch.min(pred) >= 0.0, 'Softmax distribution contains values < 0.')

        # check that values sum to one
        expected_sums = torch.ones(self.max_output_len, self.batch_size, 1)
        sums = torch.sum(pred, dim=2, keepdim=True)

        self.assertTrue(torch.allclose(expected_sums, sums), 'The softmax distribution does not sum to 1.')

    def test_same_over_batch(self):
        """
        Tests if the outputs of the 2D-LSTM are the same over the batch if the same input is fed in multiple times.
        """
        repeated_x = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        batch_x = repeated_x.expand(self.batch_size, self.max_input_len).t()

        self.lstm.eval()
        pred = self.lstm.forward(x=batch_x, y=None)     # shape (max_output_len x batch_size x vocab_size)

        pred_first = pred[:, 0, :]
        pred_expected = pred_first.expand(self.batch_size, self.max_output_len, self.vocab_size).permute(1, 0, 2)

        self.assertTrue(torch.allclose(pred, pred_expected), 'Predictions vary across same-input batch.')



