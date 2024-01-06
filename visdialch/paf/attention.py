import datetime

import math
import torch
import torch.nn as nn

from anatool import AnaLogger
from visdialch.paf.fc import MLP
from visdialch.paf.layer_norm import LayerNorm
from visdialch.paf.model_cfgs import PAFCfgs


# multi-head attention
class MHAtt(nn.Module):
    def __init__(self, paf_config: PAFCfgs, logger: AnaLogger):
        super(MHAtt, self).__init__()
        self.logger = logger
        self.paf_config = paf_config

        self.linear_v = nn.Linear(
            in_features=paf_config.HIDDEN_SIZE,
            out_features=paf_config.HIDDEN_SIZE
        )
        self.linear_k = nn.Linear(
            in_features=paf_config.HIDDEN_SIZE,
            out_features=paf_config.HIDDEN_SIZE
        )
        self.linear_q = nn.Linear(
            in_features=paf_config.HIDDEN_SIZE,
            out_features=paf_config.HIDDEN_SIZE
        )
        self.linear_merge = nn.Linear(
            in_features=paf_config.HIDDEN_SIZE,
            out_features=paf_config.HIDDEN_SIZE
        )
        self.dropout = nn.Dropout(p=paf_config.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        # v.size() (bs*rounds, proposal/len, emb_size)
        # self.logger.info(v.size())
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.paf_config.MULTI_HEAD,
            int(self.paf_config.HIDDEN_SIZE / self.paf_config.MULTI_HEAD)
        ).transpose(1, 2)
        # v.size() (bs*rounds, heads, proposal/len, emb_size)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.paf_config.MULTI_HEAD,
            int(self.paf_config.HIDDEN_SIZE / self.paf_config.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.paf_config.MULTI_HEAD,
            int(self.paf_config.HIDDEN_SIZE / self.paf_config.MULTI_HEAD)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)

        # self.logger.debug(atted.size())

        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.paf_config.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)
        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        # score.size() (bs*rounds, heads, proposal/len, proposal/len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # self.logger.debug(scores.size())

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        # --------for attention map---------
        # print(scores.shape)
        # torch.save(scores, f'attention_map_vis/data/{datetime.datetime.now()}.npy')
        # ----------------------------------

        att_map = nn.functional.softmax(scores, dim=-1)
        # att_map = self.dropout(att_map)

        # self.logger.debug(att_map.size())

        return torch.matmul(att_map, value)


# feed forward net
class FFN(nn.Module):
    def __init__(self, paf_config: PAFCfgs, logger: AnaLogger):
        super(FFN, self).__init__()

        self.logger = logger
        self.mlp = MLP(
            logger=logger,
            in_size=paf_config.HIDDEN_SIZE,
            mid_size=paf_config.FF_SIZE,
            out_size=paf_config.HIDDEN_SIZE,
            dropout_r=paf_config.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# self fusion
class SF(nn.Module):
    def __init__(self, paf_config: PAFCfgs, logger: AnaLogger):
        super(SF, self).__init__()

        self.logger = logger

        self.mh_att = MHAtt(paf_config=paf_config, logger=logger)
        self.ffn = FFN(paf_config=paf_config, logger=logger)

        self.dropout1 = nn.Dropout(p=paf_config.DROPOUT_R)
        self.norm1 = LayerNorm(size=paf_config.HIDDEN_SIZE, logger=logger)

        self.dropout2 = nn.Dropout(p=paf_config.DROPOUT_R)
        self.norm2 = LayerNorm(size=paf_config.HIDDEN_SIZE, logger=logger)

    def forward(self, y, y_mask):
        y = self.norm1(
            y + self.dropout1(
                self.mh_att(v=y, k=y, q=y, mask=y_mask)
            )
        )
        y = self.norm2(
            y + self.dropout2(
                self.ffn(y)
            )
        )
        # self.logger.debug(y.size())
        return y


# inter fusion
class IF(nn.Module):
    def __init__(self, paf_config: PAFCfgs, logger: AnaLogger):
        super(IF, self).__init__()
        self.logger = logger
        self.mh_att1 = MHAtt(paf_config=paf_config, logger=logger)
        self.mh_att2 = MHAtt(paf_config=paf_config, logger=logger)
        self.ffn = FFN(paf_config=paf_config, logger=logger)

        self.dropout1 = nn.Dropout(p=paf_config.DROPOUT_R)
        self.norm1 = LayerNorm(size=paf_config.HIDDEN_SIZE, logger=logger)

        self.dropout2 = nn.Dropout(p=paf_config.DROPOUT_R)
        self.norm2 = LayerNorm(size=paf_config.HIDDEN_SIZE, logger=logger)

        self.dropout3 = nn.Dropout(p=paf_config.DROPOUT_R)
        self.norm3 = LayerNorm(size=paf_config.HIDDEN_SIZE, logger=logger)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(
            x + self.dropout1(
                self.mh_att1(v=x, k=x, q=x, mask=x_mask)
            )
        )
        x = self.norm2(
            x + self.dropout2(
                self.mh_att2(v=y, k=y, q=x, mask=y_mask)
            )
        )
        x = self.norm3(
            x + self.dropout3(
                self.ffn(x)
            )
        )
        return x
