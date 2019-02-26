from unittest import TestCase
from unittest import skip
import torch
from model.lstm2d import LSTM2d


class LSTM2dTrainVsInferenceTest(TestCase):
    """
    Unit tests for comparing the 2d-LSTM's output in training and inference mode.
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

        device = torch.device('cpu')
        self.lstm = LSTM2d(embed_dim=self.embed_dim, state_dim_2d=self.cell_state_dim,
                           encoder_state_dim=self.encoder_state_dim, input_vocab_size=self.input_vocab_size,
                           output_vocab_size=self.output_vocab_size, device=device)

    def test_train_matches_inference(self):
        """
        Tests if the predictions in training mode match those in inference mode, if the same target tokens are used.
        """
        # random token indices of shape (max_input_len x batch_size)
        x = torch.randint(0, self.input_vocab_size, (self.max_input_len, self.batch_size), dtype=torch.long)
        x_lengths = torch.tensor([self.max_input_len]).repeat(self.batch_size)

        # first run it in inference mode, then use the generated tokens as targets for training mode and
        # then compare the results
        self.lstm.eval()
        y_pred_inference = self.lstm.forward(x=x, x_lengths=x_lengths)   # (output_seq_len x batch x output_vocab_size)
        y = torch.argmax(y_pred_inference, dim=-1)                       # (output_seq_len x batch)

        self.lstm.train()
        y_pred_train = self.lstm.forward(x=x, x_lengths=x_lengths, y=y)  # (output_seq_len x batch x output_vocab_size)

        self.assertTrue(torch.allclose(y_pred_inference, y_pred_train),
                        'Predictions vary across training vs. inference mode.')
