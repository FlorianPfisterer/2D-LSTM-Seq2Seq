import torch
import torch.nn as nn
from model.lstm2d_cell import LSTM2dCell


class LSTM2d(nn.Module):
    """
    2D-LSTM sequence-to-sequence (2D-seq2seq) model.
    Based on the paper
        "Towards two-dimensional sequence to sequence model in neural machine translation."
        Bahar, Parnia, Christopher Brix, and Hermann Ney
        arXiv preprint arXiv:1810.03975 (2018)

    Args:
        embed_dim: dimension of embedding vectors
        state_dim_2d: dimension of the hidden / cell state of the 2d-LSTM cells
        encoder_state_dim: dimension of the hidden / cell state of the bidirectional encoder LSTM
        max_input_len: maximum input sequence length
        max_output_len: maximum output sequence length
        vocab_size: size of the vocabulary (i.e. number of embedding vectors)
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

        # final softmax layer for next predicted token
        self.logits = nn.Linear(in_features=state_dim_2d, out_features=vocab_size)
        self.softmax = nn.Softmax(dim=-1)    # inputs will be of shape (max_output_len x batch x vocab_size) => last dim

        # the encoder LSTM goes over the input sequence x and provides the hidden states h_j for the 2d-LSTM
        self.encoder = nn.LSTM(input_size=embed_dim, hidden_size=encoder_state_dim, bidirectional=True)

    def forward(self, x, y):
        """
        Runs the complete forward propagation for the 2d-LSTM, using two different implementations for training
        and inference.
        Args:
            x: (max_input_len x batch) input tokens (indices in range [0, vocab_size))
            y (only if training): (max_output_len x batch) correct output tokens (indices in range [0, vocab_size))

        Returns:
            y_hat: (max_output_len x batch x vocab_size)
                predicted output sequence (softmax distribution over vocab_size)

        """
        h = self.__encoder_lstm(x)

        if self.training and y is not None:
            return self.__training_forward(h, y)
        else:
            return self.__inference_forward(h)

    def __training_forward(self, h, y):
        raise NotImplementedError()

    def __inference_forward(self, h):
        """
        Naive O(n^2) implementation of the 2d-LSTM forward pass at inference time.
        Args:
            h: (max_input_len x batch x 2*encoder_state_dim) hidden states of bidirectional encoder LSTM

        Returns:
            y: (max_output_len x batch x vocab_size) output sequence (softmax distribution over vocab_size)
        """
        batch_size = h.size()[1]
        y_i = torch.zeros(batch_size, self.embed_dim)    # TODO <start> token?

        # hidden states and cell states at previous vertical step i-1
        s_prev_i = torch.zeros(self.max_input_len, batch_size, self.state_dim_2d)
        c_prev_i = torch.zeros(self.max_input_len, batch_size, self.state_dim_2d)

        # result tensor
        y = torch.empty(self.max_output_len, batch_size, self.vocab_size)

        # go through each decoder output step
        for i in range(self.max_output_len):
            # initialize previous horizontal hidden state and cell state
            s_prev_hor = torch.zeros(batch_size, self.state_dim_2d)
            c_prev_hor = torch.zeros(batch_size, self.state_dim_2d)

            for j in range(self.max_input_len):
                # input to 2d-cell is concatenation of encoder hidden state h_j and last generated token y_i
                h_j = h[j, :, :]
                x_j = torch.cat([y_i, h_j], dim=1)

                s_prev_ver = s_prev_i[j, :, :]
                c_prev_ver = c_prev_i[j, :, :]

                c_prev_hor, s_prev_hor = self.cell2d.forward(x_j, s_prev_hor, s_prev_ver, c_prev_hor, c_prev_ver)

            # obtain next predicted token
            s_Ji = s_prev_hor   # final hidden state at this vertical step i
            y_pred_i = self.logits.forward(s_prev_hor)
            y_pred_i = self.softmax.forward(y_pred_i)
            y[i, :, :] = y_pred_i

            # next generated token embedding (TODO beam seach?)
            y_i = self.embedding.forward(torch.argmax(y_pred_i, dim=1))

        return y

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

