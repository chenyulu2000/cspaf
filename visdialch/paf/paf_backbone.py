import time

import torch
import torch.nn as nn

from anatool import AnaLogger, AnaArgParser
from visdialch.paf.attention import SF, IF
from visdialch.paf.model_cfgs import PAFCfgs


class PAFBackbone(nn.Module):
    def __init__(self, paf_config: PAFCfgs, logger: AnaLogger):
        super(PAFBackbone, self).__init__()
        self.logger = logger

        self.enc_list = nn.ModuleList(
            [SF(paf_config=paf_config, logger=logger) for _ in range(paf_config.LAYER)]
        )
        self.dec_list = nn.ModuleList(
            [IF(paf_config=paf_config, logger=logger) for _ in range(paf_config.LAYER)]
        )
        self.USING_PAF = paf_config.USING_PAF

    def forward(self, y, x, y_mask, x_mask):
        """
        :param y: lang (bs*bounds, seq_len, emb_size)
        :param x: img (bs*bounds, proposals, emb_size)
        :param y_mask: (bs*bounds, seq_len)
        :param x_mask: (bs*rounds, proposals)
        """
        if self.USING_PAF:
            for idx in range(len(self.enc_list)):
                enc = self.enc_list[idx]
                dec = self.dec_list[idx]

                y = enc(y, y_mask)
                x = dec(x, y, x_mask, y_mask)
        else:
            for enc in self.enc_list:
                y = enc(y, y_mask)

            for dec in self.dec_list:
                x = dec(x, y, x_mask, y_mask)
        return y, x


if __name__ == '__main__':
    parser = AnaArgParser()
    opt = parser.cfg
    logger = AnaLogger(opt.exp_dir)
    start = time.time()
    bb = PAFBackbone(
        logger=logger,
        paf_config=PAFCfgs()
    )
    a, b = bb(
        x=torch.rand(1280, 16, 512),
        y=torch.rand(1280, 16, 512),
        x_mask=torch.ones(1280, 1, 1, 16),
        y_mask=torch.ones(1280, 1, 1, 16)
    )
    torch.sum(a).backward(retain_graph=True)
    torch.sum(b).backward(retain_graph=True)
    end = time.time()
    logger.info(end - start)
