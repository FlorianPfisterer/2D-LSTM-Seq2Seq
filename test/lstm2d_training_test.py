from unittest import TestCase
from unittest import skip
import torch
from model.lstm2d import LSTM2d


class LSTM2dTrainingTest(TestCase):
    """
    Unit tests for the 2D-LSTM in training mode.
    """
    embed_dim = 50
    encoder_state_dim = 20
    cell_state_dim = 25

    batch_size = 42

    max_input_len = 4
    max_output_len = 5

    input_vocab_size = 4
    output_vocab_size = 3

    def setUp(self):
        torch.manual_seed(42)

        self.lstm = LSTM2d(embed_dim=self.embed_dim, state_dim_2d=self.cell_state_dim,
                           encoder_state_dim=self.encoder_state_dim, input_vocab_size=self.input_vocab_size,
                           output_vocab_size=self.output_vocab_size)

    def test_dimensions(self):
        """
        Tests if the input and output dimensions of the 2D-LSTM are as expected.
        """
        # random token indices of shape (max_input_len x batch_size)
        sample_x = torch.randint(0, self.input_vocab_size, (self.max_input_len, self.batch_size), dtype=torch.long)
        sample_y = torch.randint(0, self.output_vocab_size, (self.max_output_len, self.batch_size), dtype=torch.long)

        # toy training
        self.lstm.train()
        pred = self.lstm.forward(x=sample_x, y=sample_y)

        pred_shape = list(pred.shape)
        self.assertEqual(pred_shape, [self.max_output_len, self.batch_size, self.output_vocab_size],
                         'The predictions have an unexpected shape.')

    @skip("the 2D-LSTM now outputs the logits directly, since the cross-entropy loss takes care of the softmax")
    def test_valid_softmax(self):
        """
        Tests if the output predictions of the 2D-LSTM form a valid softmax distribution over the vocabulary, i.e.
        the elements are in [0, 1] and sum to 1.
        """
        sample_x = torch.randint(0, self.input_vocab_size, (self.max_input_len, self.batch_size), dtype=torch.long)
        sample_y = torch.randint(0, self.output_vocab_size, (self.max_output_len, self.batch_size), dtype=torch.long)

        self.lstm.train()
        pred = self.lstm.forward(x=sample_x, y=sample_y)    # shape (max_output_len x batch_size x vocab_size)

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
        repeated_x = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        repeated_y = torch.tensor([0, 2, 0, 1, 2], dtype=torch.long)
        batch_x = repeated_x.expand(self.batch_size, self.max_input_len).t()
        batch_y = repeated_y.expand(self.batch_size, self.max_output_len).t()

        self.lstm.train()
        pred = self.lstm.forward(x=batch_x, y=batch_y)     # shape (max_output_len x batch_size x vocab_size)

        pred_first = pred[:, 0, :]
        pred_expected = pred_first.expand(self.batch_size, self.max_output_len, self.output_vocab_size).permute(1, 0, 2)

        self.assertTrue(torch.allclose(pred, pred_expected), 'Predictions vary across same-input batch.')
