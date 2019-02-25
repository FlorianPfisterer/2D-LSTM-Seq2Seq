import torch.nn as nn


class LSTM2dCell(nn.Module):
    """
    A 2d-LSTM Cell that computes it's hidden state and cell state based on
        - an input x
        - the previous horizontal hidden and cell state
        - the previous vertical hidden and cell state

    Args:
        input_dim: the input dimension (i.e. second dimension of x)
        state_dim: dimension of the hidden and cell state of this LSTM unit
        device: the device (CPU / GPU) to run all computations on / store tensors on
    """

    def __init__(self, input_dim, state_dim, device):
        super(LSTM2dCell, self).__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.device = device

        # input to state
        self.W_x = nn.Linear(self.input_dim, self.state_dim * 5).to(self.device)
        # previous horizontal hidden state to state
        self.W_hor = nn.Linear(self.state_dim, self.state_dim * 5).to(self.device)
        # previous vertical hidden state to state
        self.W_ver = nn.Linear(self.state_dim, self.state_dim * 5).to(self.device)

    def forward(self, x, s_prev_hor, s_prev_ver, c_prev_hor, c_prev_ver):
        """
        Forward pass of the 2d-LSTM Cell at horizontal step j and vertical step i (to compute c_ji and s_ji)

        Args:
            x: (batch x input_dim) input at horizontal step j
            s_prev_hor: (batch x state_dim) hidden state of cell at previous horizontal step j-1, same vertical step i
            s_prev_ver: (batch x state_dim) hidden state of cell at previous vertical step i-1, same horizontal step j
            c_prev_hor: (batch x state_dim) cell state of cell at previous horizontal step j-1, same vertical step i
            c_prev_ver: (batch x state_dim) cell state of cell at previous vertical step i-1, same horizontal step j

        Returns:
            c: (batch x state_dim) next cell state (c_ji)
            s: (batch x state_dim) next hidden state (s_ji)
        """
        pre_activation = self.W_x(x) + self.W_hor(s_prev_hor) + self.W_ver(s_prev_ver)
        gates = pre_activation[:, :4*self.state_dim].sigmoid()

        # retrieve input, forget, output and lambda gate from gates
        i = gates[:, 0*self.state_dim:1*self.state_dim]
        f = gates[:, 1*self.state_dim:2*self.state_dim]
        o = gates[:, 2*self.state_dim:3*self.state_dim]
        l = gates[:, 3*self.state_dim:4*self.state_dim]

        c_candidate = pre_activation[:, 4*self.state_dim:].tanh()
        c = f * (l * c_prev_hor + (1 - l) * c_prev_ver) + c_candidate * i
        s = c.tanh() * o

        return c, s



