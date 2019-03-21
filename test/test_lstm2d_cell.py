from unittest import TestCase
import torch
from model.lstm2d_cell import LSTM2dCell


class LSTM2dCellTest(TestCase):
    """
    Unit tests for the 2D-LSTM cell.
    """
    embed_dim = 50
    encoder_state_dim = 20
    input_dim = 2 * encoder_state_dim + embed_dim
    cell_state_dim = 25
    batch_size = 42

    def setUp(self):
        torch.manual_seed(42)

        self.x_j = torch.randn(self.batch_size, self.input_dim)
        self.s_prev_hor = torch.randn(self.batch_size, self.cell_state_dim)
        self.s_prev_ver = torch.randn(self.batch_size, self.cell_state_dim)
        self.c_prev_hor = torch.randn(self.batch_size, self.cell_state_dim)
        self.c_prev_ver = torch.randn(self.batch_size, self.cell_state_dim)

        self.device = torch.device('cpu')

    def test_dimensions(self):
        """
        Tests if the input and output dimensions of the cell are as expected.
        """
        cell = LSTM2dCell(self.input_dim, self.cell_state_dim, self.device)
        c_ji, s_ji = cell.forward(x=self.x_j, s_prev_hor=self.s_prev_hor, s_prev_ver=self.s_prev_ver,
                                  c_prev_hor=self.c_prev_hor, c_prev_ver=self.c_prev_ver)

        c_shape = list(c_ji.shape)
        s_shape = list(s_ji.shape)

        self.assertEqual(c_shape, [self.batch_size, self.cell_state_dim], 'Next cell state has unexpected shape')
        self.assertEqual(s_shape, [self.batch_size, self.cell_state_dim], 'Next hidden state has unexpected shape')

    def test_same_over_batch(self):
        """
        Tests if the outputs of the cell are the same over the batch if the same input is fed in multiple times.
        """
        toy_input_dim = 4
        toy_batch_size = 7
        toy_state_dim = 3

        # create toy values and repeat them over the batch
        toy_x = torch.Tensor([1.5, 4.2, 3.1415, 2.71]).expand(toy_batch_size, toy_input_dim)

        toy_s_prev_hor = torch.Tensor([-.4, 1.2, 42.195]).expand(toy_batch_size, toy_state_dim)
        toy_s_prev_ver = torch.Tensor([2.3, 7.12, -3.14]).expand(toy_batch_size, toy_state_dim)

        toy_c_prev_hor = torch.Tensor([-10.1, 4.5, -0.1]).expand(toy_batch_size, toy_state_dim)
        toy_c_prev_ver = torch.Tensor([17, 1.001, -2.23]).expand(toy_batch_size, toy_state_dim)

        cell = LSTM2dCell(toy_input_dim, toy_state_dim, self.device)
        c, s = cell.forward(x=toy_x, s_prev_hor=toy_s_prev_hor, s_prev_ver=toy_s_prev_ver,
                            c_prev_hor=toy_c_prev_hor, c_prev_ver=toy_c_prev_ver)

        # check if the cell and hidden state are the same across the whole batch
        c_first = c[0, :]
        repeated_c_first = c_first.expand(toy_batch_size, c_first.shape[-1])
        self.assertTrue(repeated_c_first.allclose(c), 'Next cell state varies across same-input batch')

        s_first = s[0, :]
        repeated_s_first = s_first.expand(toy_batch_size, s_first.shape[-1])
        self.assertTrue(repeated_s_first.allclose(s), 'Next hidden state varies across same-input batch')

