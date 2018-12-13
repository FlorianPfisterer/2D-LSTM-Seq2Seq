import torch
import torch.nn as nn
from model.lstm2d_cell import LSTM2dCell


class LSTM2d(nn.Module):
    """
    2d-LSTM network

    Args:
        embed_dim:
        state_dim_2d:
        encoder_state_dim:
        max_input_len:
        max_output_len:
        vocab_size:
    """
    def __init__(self, embed_dim, state_dim_2d, encoder_state_dim, max_input_len, max_output_len, vocab_size):
        super(LSTM2d, self).__init__()

        self.embed_dim = embed_dim
        self.state_dim_2d = state_dim_2d
        self.encoder_state_dim = encoder_state_dim
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

        # input to the 2d-cell is a concatenation of the hidden encoder states h_j and the embedded output tokens y_i-1
        cell_input_dim = 2*encoder_state_dim + embed_dim    # 2*encoder_state_dim since it's bidirectional
        self.cell2d = LSTM2dCell(cell_input_dim, state_dim_2d)

        # the encoder LSTM goes over the input sequence x and provides the hidden states h_j for the 2d-LSTM
        self.encoder = nn.LSTM(input_size=embed_dim, hidden_size=encoder_state_dim, bidirectional=True)

    def forward(self, x, y):
        h = self.__encoder_lstm(x)

        if self.training and y is not None:
            return self.__training_forward(x, y)
        else:
            return self.__inference_forward(h)

    def __training_forward(self, h, y):
        raise NotImplementedError()

    def __inference_forward(self, h):
        """
        Naive O(n^2) implementation of the 2d-LSTM forward propagation
        Args:
            h: (max_input_len x batch x 2*encoder_state_dim) hidden states of bidirectional encoder LSTM

        Returns:
            y: (max_output_len x batch x vocab_size) output sequence (softmax distribution over vocab_size)
        """
        raise NotImplementedError()

    def __encoder_lstm(self, x):
        """
        Runs the bidirectional encoder LSTM on the input sequence to obtain the hidden states h_j.
        Args:
            x: (max_input_len x batch) input tokens

        Returns:
            h: (max_input_len x batch x 2*encoder_state_dim) hidden states of bidirectional encoder LSTM
        """

        embedded_x = self.embedding.forward(x)      # (max_input_len x batch x embed_dim)
        h, _ = self.encoder.forward(embedded_x)     # (max_input_len x batch x 2*encoder_state_dim)

        return h

