import torch
import torch.nn as nn

class LSTM2d(nn.Module):
    def __init__(self, input_dim, state_dim):
        super(LSTM2d, self).__init__()

        self.input_dim = input_dim
        self.state_dim = state_dim

        # TODO