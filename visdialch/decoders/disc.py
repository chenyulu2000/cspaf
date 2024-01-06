import torch
import torch.nn as nn

from anatool import AnaArgParser, AnaLogger
from visdialch.data.vocabulary import Vocabulary
from visdialch.utils.dynamic_rnn import DynamicRNN


class DiscriminativeDecoder(nn.Module):
    def __init__(self, opt, logger: AnaLogger, vocabulary):
        super(DiscriminativeDecoder, self).__init__()
        self.opt = opt
        self.logger = logger

        self.word_embed = nn.Embedding(
            num_embeddings=len(vocabulary),
            embedding_dim=opt.word_embedding_size,
            padding_idx=vocabulary.PAD_INDEX
        )
        self.option_rnn = nn.LSTM(
            opt.word_embedding_size,
            opt.lstm_hidden_size,
            opt.lstm_num_layers,
            batch_first=True,
            dropout=opt.dropout
        )

        self.option_rnn = DynamicRNN(logger=logger, rnn_model=self.option_rnn)

    def forward(self, encoder_output, batch):
        """Given `encoder_output` + candidate option sequences, predict a score
        for each option sequence.

        Parameters
        ----------
        encoder_output: torch.Tensor
            Output from the encoder through its forward pass.
            (batch_size, num_rounds, lstm_hidden_size)
        batch:
        """
        options = batch['opt']
        batch_size, num_rounds, num_options, max_sequence_length = options.size()
        options = options.view(batch_size * num_rounds * num_options, max_sequence_length)
        options_length = batch['opt_len']
        options_length = options_length.view(batch_size * num_rounds * num_options)

        # pick options with non-zero length (relevant for test split).
        # nonzero_options_length_indices = torch.nonzero(input=options_length, as_tuple=False).squeeze()
        nonzero_options_length_indices = options_length.nonzero().squeeze()
        nonzero_options_length = options_length[nonzero_options_length_indices]
        nonzero_options = options[nonzero_options_length_indices]

        # shape: (batch_size * num_rounds * num_options, max_sequence_length,
        #         word_embedding_size)
        # for test split, shape: (batch_size * 1, num_options, lstm_hidden_size)
        nonzero_options_embed = self.word_embed(nonzero_options)
        _, (nonzero_options_embed, _) = self.option_rnn(
            nonzero_options_embed, nonzero_options_length
        )

        options_embed = torch.zeros(
            batch_size * num_rounds * num_options,
            nonzero_options_embed.size(-1),
            device=nonzero_options_embed.device
        )
        options_embed[nonzero_options_length_indices] = nonzero_options_embed

        # reapeat encoder output for every option.
        # shape: (batch_size, num_rounds, num_options, max_sequence_length)
        encoder_output = encoder_output.unsqueeze(2).repeat(
            1, 1, num_options, 1
        )

        # shape now same as 'options', can calculate dot producat similarity.
        # shape: (batch_size * num_rounds * num_options, lstm_hidden_state)
        encoder_output = encoder_output.view(
            batch_size * num_rounds * num_options,
            self.opt.lstm_hidden_size
        )

        # shape: (batch_size * num_rounds *num_options)
        scores = torch.sum(options_embed * encoder_output, 1)
        # shape: (batch_size, num_rounds, num_options)
        scores = scores.view(batch_size, num_rounds, num_options)
        return scores


if __name__ == '__main__':
    parser = AnaArgParser()
    opt = parser.cfg
    logger = AnaLogger(opt.exp_dir)
    dd = DiscriminativeDecoder(
        opt=opt,
        logger=logger,
        vocabulary=Vocabulary(
            word_counts_path=opt.word_counts_json,
            min_count=opt.vocab_min_count,
            logger=logger
        )
    )
    dd(
        torch.rand(32, 10, 512).long(),
        {
            'opt': torch.rand(32, 10, 100, 20).long(),
            'opt_len': torch.ones(32, 10, 100).long(),
        }
    )
