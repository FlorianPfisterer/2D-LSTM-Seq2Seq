import torch.nn as nn


class LSTM2dCell(nn.Module):
    def __init__(self, input_dim, state_dim):
        super(LSTM2dCell, self).__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim

        # input to state
        self.W = nn.Linear(self.input_dim, self.state_dim * 5)
        # previous horizontal hidden state to state
        self.U = nn.Linear(self.state_dim, self.state_dim * 5)
        # previous vertical hidden state to state
        self.V = nn.Linear(self.state_dim, self.state_dim * 5)

    def forward(self, xj, s_horizontal, s_vertical, c_horizontal, c_vertical):
        """
        Forward pass of the 2d-LSTM Cell at horizontal step j and vertical step i (to compute c_ji and s_ji)

        :param xj: input at horizontal step j
        :param s_horizontal: hidden state of cell at previous horizontal step j-1, same vertical step i
        :param s_vertical: hidden state of cell at previous vertical step i-1, same horizontal step j
        :param c_horizontal: cell state of cell at previous horizontal step j-1, same vertical step i
        :param c_vertical: cell state of cell at previous vertical step i-1, same horizontal step j
        :return:
            c_ji: next cell state
            s_ji: next hidden state
        """
        pre_activation = self.W(xj) + self.U(s_horizontal) + self.V(s_vertical)
        gates = pre_activation[:, :4*self.state_dim].sigmoid()

        # retrieve input, forget, output and lambda gate from gates
        i = gates[:, 0*self.state_dim:1*self.state_dim]
        f = gates[:, 1*self.state_dim:2*self.state_dim]
        o = gates[:, 2*self.state_dim:3*self.state_dim]
        l = gates[:, 3*self.state_dim:4*self.state_dim]

        c_candidate = pre_activation[:, 4*self.state_dim:].tanh()
        c_ji = f * (l * c_horizontal + (1 - l) * c_vertical) + c_candidate * i
        s_ji = c_ji.tanh() * o

        return c_ji, s_ji



