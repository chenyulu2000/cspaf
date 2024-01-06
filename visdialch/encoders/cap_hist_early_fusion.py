import random

import torch
import torch.nn as nn

from anatool import AnaLogger, AnaArgParser
from visdialch.data.vocabulary import Vocabulary
from visdialch.paf.make_mask import make_mask
from visdialch.paf.model_cfgs import PAFCfgs
from visdialch.paf.paf import PAF
from visdialch.utils.dynamic_rnn import DynamicRNN


class CaptionHistoryEarlyFusionEncoder(nn.Module):
    def __init__(self, opt, logger: AnaLogger, vocabulary):
        super().__init__()
        self.logger = logger
        self.opt = opt
        self.vocabulary = vocabulary

        self.paf_config = PAFCfgs()
        # self.hist_ques_fusion=
        self.hist_ques_mcan_net = PAF(
            paf_config=self.paf_config,
            logger=self.logger,
            answer_size=self.opt.lstm_hidden_size,
        )
        self.img_ques_mcan_net = PAF(
            paf_config=self.paf_config,
            logger=self.logger,
            answer_size=self.opt.lstm_hidden_size,
        )

        self.word_embed = nn.Embedding(
            num_embeddings=len(self.vocabulary),
            embedding_dim=self.opt.word_embedding_size,
            padding_idx=self.vocabulary.PAD_INDEX,
        )

        self.ques_rnn = nn.LSTM(
            input_size=self.opt.word_embedding_size,
            hidden_size=self.opt.lstm_hidden_size,
            num_layers=self.opt.lstm_num_layers,
            batch_first=True
        )
        self.ques_rnn = DynamicRNN(logger=logger, rnn_model=self.ques_rnn)

        self.hist_rnn = nn.LSTM(
            input_size=self.opt.word_embedding_size,
            hidden_size=self.opt.lstm_hidden_size,
            num_layers=self.opt.lstm_num_layers,
            batch_first=True,
            dropout=self.opt.dropout
        )

        self.hist_rnn = DynamicRNN(logger=logger, rnn_model=self.hist_rnn)

        # project image features to lstm_hidden_size for computing attention
        self.image_features_projection = nn.Linear(
            in_features=opt.img_feature_size,
            out_features=opt.lstm_hidden_size
        )

        self.fusion = nn.Linear(
            in_features=2 * opt.lstm_hidden_size,
            out_features=opt.lstm_hidden_size
        )

        self.dropout = nn.Dropout(p=opt.dropout)

        nn.init.kaiming_uniform_(self.image_features_projection.weight)
        nn.init.constant_(self.image_features_projection.bias, 0)
        nn.init.kaiming_uniform_(self.fusion.weight)
        nn.init.constant_(self.fusion.bias, 0)

    def forward(self, batch, cfq_batch=False, cfi_batch=False):
        # shape: (batch_size, num_proposals, img_feature_size)
        img = batch['img_feat']
        # shape: (batch_size, max_cap_sequence_length)
        cap = batch['cap']
        cap_len = batch['cap_len']
        # shape: (batch_size, 10, max_sequence_length)
        # num_rounds = 10, even for test (padded dialog rounds at the end)
        ques = batch['ques']
        ques_len = batch['ques_len']
        # shape: (batch_size, 10, max_sequence_length * 2)
        # shape: (batch_size, 10, max_sequence_length * 2) if concat qa * 9 rounds
        hist = batch['hist']
        hist_len = batch['hist_len']

        batch_size, num_rounds, combined_max_sequence_length = hist.size()
        _, _, max_sequence_length = ques.size()
        _, max_cap_sequence_length = cap.size()

        if cfq_batch:
            index_cfq = random.sample(range(0, batch_size), batch_size)
            ques, ques_len = ques[index_cfq], ques_len[index_cfq]

        if cfi_batch:
            index_cfi = random.sample(range(0, batch_size), batch_size)
            img = img[index_cfi]

        # project down image features and ready for attention
        # shape: (batch_size, num_proposals, lstm_hidden_size)
        projected_image_features = self.image_features_projection(img)

        # maybe it is redundant
        projected_image_features = (
            projected_image_features.view(
                batch_size, 1, -1, self.opt.lstm_hidden_size
            )
            .repeat(1, num_rounds, 1, 1)
            .view(batch_size * num_rounds, -1, self.opt.lstm_hidden_size)
        )

        image_features = projected_image_features

        ques = ques.view(batch_size * num_rounds, max_sequence_length)
        lang_ques_feat_mask = make_mask(ques.unsqueeze(2))
        ques = ques.squeeze(1)
        ques_embed = self.word_embed(ques.long())

        ques_embed_per_word, (ques_embed, _) = self.ques_rnn(ques_embed, ques_len)

        # shape (batch_size * num_rounds, lstm_hidden_size)
        img_ques_mcan_embedding = self.img_ques_mcan_net(
            image_features,
            ques_embed_per_word,
            lang_ques_feat_mask
        )
        hist_len = hist_len.unsqueeze(dim=-1).view(batch_size, num_rounds, -1)
        cap_len = cap_len.unsqueeze(dim=-1).unsqueeze(dim=-1).view(batch_size, 1, 1)
        hist = hist[:, :-1, :]
        hist_len = hist_len[:, :-1, :]
        cap = cap.unsqueeze(dim=1).view(batch_size, 1, -1)
        hist = torch.cat([cap, hist], dim=1)
        hist_len = torch.cat([cap_len, hist_len], dim=1)
        hist = hist.view(batch_size * num_rounds, -1)
        hist_embed = self.word_embed(hist.long())
        hist_embed_per_word, (hist_embed, _) = self.hist_rnn(hist_embed, hist_len)

        # shape (batch_size * num_rounds, lstm_hidden_size)
        hist_ques_mcan_embedding = self.hist_ques_mcan_net(
            hist_embed_per_word,
            ques_embed_per_word,
            lang_ques_feat_mask
        )

        fused_vector = torch.cat((img_ques_mcan_embedding, hist_ques_mcan_embedding), dim=1)
        fused_vector = self.dropout(fused_vector)

        # shape (batch_size * num_rounds, lstm_hidden_size)
        fused_embedding = self.fusion(fused_vector)
        return fused_embedding.view(batch_size, num_rounds, -1)


if __name__ == '__main__':
    parser = AnaArgParser()
    opt = parser.cfg
    logger = AnaLogger(opt.exp_dir)
    icfe = CaptionHistoryEarlyFusionEncoder(
        opt=opt,
        logger=logger,
        vocabulary=Vocabulary(
            word_counts_path=opt.word_counts_json,
            min_count=opt.vocab_min_count,
            logger=logger
        )
    )
    total = sum([param.nelement() for param in icfe.parameters()])
    logger.info("Number of parameter: %.2fM" % (total / 1e6))
    icfe(batch={
        'img_feat': torch.rand(32, 36, 2048, dtype=torch.float),
        'cap': torch.rand(32, 40, dtype=torch.float),
        'cap_len': torch.ones(32, 1, 1, dtype=torch.int),
        'ques': torch.rand(32, 10, 20, dtype=torch.float),
        'ques_len': torch.ones(32, 10, 1, dtype=torch.int),
        'hist': torch.rand(32, 10, 40, dtype=torch.float),
        'hist_len': torch.ones(32, 10, 1, dtype=torch.int),
    })
