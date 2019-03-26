from unittest import TestCase
from unittest import skip
import torch
from model.lstm2d import LSTM2d


class LSTM2dTrainVsInferenceTest(TestCase):
    """
    Unit tests for comparing the 2d-LSTM's output in training and inference mode.
    """
    embed_dim = 10
    encoder_state_dim = 16
    cell_state_dim = 32

    batch_size = 20

    max_input_len = 40
    max_output_len = 50

    input_vocab_size = 100
    output_vocab_size = 100

    pad_token = 0

    def setUp(self):
        torch.manual_seed(42)

        device = torch.device('cpu')
        self.lstm = LSTM2d(embed_dim=self.embed_dim, state_dim_2d=self.cell_state_dim,
                           encoder_state_dim=self.encoder_state_dim, input_vocab_size=self.input_vocab_size,
                           output_vocab_size=self.output_vocab_size, max_output_len=self.max_output_len, device=device,
                           pad_token=self.pad_token, dropout_p=0)

    def test_train_matches_inference(self):
        """
        Tests if the predictions in training mode match those in inference mode, if the same target tokens are used.
        """
        # random token indices of shape (max_input_len x batch_size)
        x = torch.randint(0, self.input_vocab_size, (self.max_input_len, self.batch_size), dtype=torch.long)
        x_lengths = torch.tensor([self.max_input_len]).repeat(self.batch_size)

        self.__assert_same_results(x, x_lengths)

    def test_train_matches_inference_with_padding(self):
        """
        Tests if the predictions in training mode match those in inference mode, if the same target tokens are used,
        with padding (i.e. the input sequences are not all of the same length)
        """
        # random token indices of shape (max_input_len x batch_size)
        x = torch.randint(3, self.input_vocab_size, (self.max_input_len, self.batch_size), dtype=torch.long)

        # use random lengths (simulate "padding")
        x_lengths = torch.randint(5, self.max_input_len, (self.batch_size, 1), dtype=torch.long).view(-1)
        x_lengths, _ = torch.sort(x_lengths, descending=True)
        for (i, length) in enumerate(x_lengths):
            x[length:, i] = self.pad_token

        self.__assert_same_results(x, x_lengths)

    def __assert_same_results(self, x, x_lengths):
        """
        Makes sure the predictions of the 2d-LSTM are the same in inference and training mode for the given inputs if
        the same target tokens are used.
        """
        self.lstm.eval()
        # first run it in inference mode, then use the generated tokens as targets for training mode and
        # then compare the results

        y_pred_inference = self.lstm.predict(x=x, x_lengths=x_lengths)  # (output_seq_len x batch x output_vocab_size)
        y = torch.argmax(y_pred_inference, dim=-1)  # (output_seq_len x batch)

        y_pred_train = self.lstm.forward(x=x, x_lengths=x_lengths, y=y)  # (output_seq_len x batch x output_vocab_size)

        self.assertTrue(torch.allclose(y_pred_inference, y_pred_train, atol=1e-04),
                        'Predictions vary across training vs. inference mode.')