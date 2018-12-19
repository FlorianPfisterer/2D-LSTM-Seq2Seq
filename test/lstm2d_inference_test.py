from unittest import TestCase, skip
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

    # in inference mode, the output length is always assumed to be the same as the input length
    input_seq_len = 4
    output_seq_len = input_seq_len

    input_vocab_size = 3
    output_vocab_size = 5

    def setUp(self):
        torch.manual_seed(42)

        self.lstm = LSTM2d(embed_dim=self.embed_dim, state_dim_2d=self.cell_state_dim,
                           encoder_state_dim=self.encoder_state_dim, input_vocab_size=self.input_vocab_size,
                           output_vocab_size=self.output_vocab_size)

    def test_dimensions(self):
        """
        Tests if the input and output dimensions of the 2D-LSTM are as expected.
        """
        # random token indices of shape (input_seq_len x batch_size)
        sample_x = torch.randint(0, self.input_vocab_size, (self.input_seq_len, self.batch_size), dtype=torch.long)

        # toy inference
        self.lstm.eval()
        pred = self.lstm.forward(x=sample_x, y=None)

        pred_shape = list(pred.shape)
        self.assertEqual(pred_shape, [self.output_seq_len, self.batch_size, self.output_vocab_size],
                         'The predictions have an unexpected shape.')

    @skip("the 2D-LSTM now outputs the logits directly, since the cross-entropy loss takes care of the softmax")
    def test_valid_softmax(self):
        """
        Tests if the output predictions of the 2D-LSTM form a valid softmax distribution over the vocabulary, i.e.
        the elements are in [0, 1] and sum to 1.
        """
        sample_x = torch.randint(0, self.input_vocab_size, (self.input_seq_len, self.batch_size), dtype=torch.long)

        self.lstm.eval()
        pred = self.lstm.forward(x=sample_x, y=None)    # shape (output_seq_len x batch_size x vocab_size)

        # check [0, 1] range
        self.assertTrue(torch.max(pred) <= 1.0, 'Softmax distribution contains values > 1.')
        self.assertTrue(torch.min(pred) >= 0.0, 'Softmax distribution contains values < 0.')

        # check that values sum to one
        expected_sums = torch.ones(self.output_seq_len, self.batch_size, 1)
        sums = torch.sum(pred, dim=2, keepdim=True)

        self.assertTrue(torch.allclose(expected_sums, sums), 'The softmax distribution does not sum to 1.')

    def test_same_over_batch(self):
        """
        Tests if the outputs of the 2D-LSTM are the same over the batch if the same input is fed in multiple times.
        """
        repeated_x = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        batch_x = repeated_x.expand(self.batch_size, self.input_seq_len).t()

        self.lstm.eval()
        pred = self.lstm.forward(x=batch_x, y=None)     # shape (output_seq_len x batch_size x vocab_size)

        pred_first = pred[:, 0, :]
        pred_expected = pred_first.expand(self.batch_size, self.output_seq_len, self.output_vocab_size).permute(1, 0, 2)

        self.assertTrue(torch.allclose(pred, pred_expected), 'Predictions vary across same-input batch.')
