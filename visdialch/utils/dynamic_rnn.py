import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from anatool import AnaLogger, AnaArgParser


class DynamicRNN(nn.Module):
    # support LSTM and GRU cells.
    def __init__(self, logger: AnaLogger, rnn_model, rnn_type='LSTM', bi_enc=False):
        super(DynamicRNN, self).__init__()
        self.logger = logger
        self.rnn_model = rnn_model
        self.rnn_type = rnn_type
        self.bi_enc = bi_enc

    def forward(self, seq_input, seq_lens, initial_state=None):
        """A wrapper over pytorch's rnn to handle sequences of variable length.

        Arguments
        ---------
        seq_input : torch.Tensor
            Input sequence tensor (padded) for RNN model.
            Shape: (batch_size, max_sequence_length, embed_size)
        seq_lens : torch.LongTensor
            Length of sequences (b, )
        initial_state : torch.Tensor
            Initial (hidden, cell) states of RNN model.

        Returns
        -------
            A single tensor of shape (batch_size, rnn_hidden_size) corresponding
            to the outputs of the RNN model at the last time step of each input
            sequence.
        """
        max_sequence_length = seq_input.size(1)
        sorted_len, fwd_order, bwd_order = self._get_sorted_order(seq_lens)
        sorted_seq_input = seq_input.index_select(0, fwd_order)
        packed_seq_input = pack_padded_sequence(
            input=sorted_seq_input,
            lengths=sorted_len.cpu(),
            batch_first=True
        )
        if initial_state is not None:
            hx = initial_state
            assert hx[0].size(0) == self.rnn_model.num_layers
        else:
            hx = None

        self.rnn_model.flatten_parameters()

        if self.rnn_type == 'LSTM':
            outputs, (h_n, c_n) = self.rnn_model(packed_seq_input, hx)
        else:
            # GRU cell doesn't have c_n.
            outputs, h_n = self.rnn_model(packed_seq_input, hx)
            c_n = None
        if self.bi_enc:
            dim_0 = h_n.size(0)
            h_n = torch.cat([h_n[0:dim_0:2], h_n[1:dim_0:2]], 2)
        h_n = h_n[-1].index_select(dim=0, index=bwd_order)
        if self.rnn_type == 'LSTM':
            c_n = c_n[-1].index_select(dim=0, index=bwd_order)

        outputs_tuple = pad_packed_sequence(
            sequence=outputs,
            batch_first=True,
            total_length=max_sequence_length
        )
        outputs = outputs_tuple[0].index_select(dim=0, index=bwd_order)
        return outputs, (h_n, c_n)

    @staticmethod
    def _get_sorted_order(lens):
        sorted_len, fwd_order = torch.sort(lens.contiguous().view(-1), 0, descending=True)
        _, bwd_order = torch.sort(fwd_order)
        return sorted_len, fwd_order, bwd_order


if __name__ == '__main__':
    parser = AnaArgParser()
    opt = parser.cfg
    logger = AnaLogger(opt.exp_dir)
    dr = DynamicRNN(
        logger=logger,
        rnn_model=nn.LSTM(
            input_size=opt.word_embedding_size,
            hidden_size=opt.lstm_hidden_size,
            num_layers=opt.lstm_num_layers,
            batch_first=True
        ),
        rnn_type='LSTM',
    )
    dr.forward(
        seq_input=torch.rand(32, 20, 300, dtype=torch.float),
        seq_lens=torch.ones(32),
    )
