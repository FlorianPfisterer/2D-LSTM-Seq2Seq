from unittest import TestCase
import torch
from model.lstm2d import LSTM2d


class LSTM2dTrainingTest(TestCase):
    """
    Unit tests for the 2D-LSTM in with inhomogeneous sequence lengths (padding).
    """
    embed_dim = 50
    encoder_state_dim = 20
    cell_state_dim = 25

    input_vocab_size = 10
    output_vocab_size = 10

    pad_token = 0
    bos_token = 1
    eos_token = 2

    def setUp(self):
        torch.manual_seed(42)

        device = torch.device('cpu')
        self.lstm = LSTM2d(embed_dim=self.embed_dim, state_dim_2d=self.cell_state_dim,
                           encoder_state_dim=self.encoder_state_dim, input_vocab_size=self.input_vocab_size,
                           output_vocab_size=self.output_vocab_size, device=device, pad_token=self.pad_token,
                           bos_token=self.bos_token, eos_token=self.eos_token)

    def test_toy_batch(self):
        x = torch.tensor([[4, 6, 7],
                          [3, 5, 9],
                          [8, 3, 0],
                          [8, 0, 0],
                          [3, 0, 0]], dtype=torch.long)
        x_lengths = torch.tensor([5, 3, 2], dtype=torch.long)

        self.lstm.eval()
        y_pred = self.lstm.forward(x, x_lengths)

        # TODO how to check correct handling
