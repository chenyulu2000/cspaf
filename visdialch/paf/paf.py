import torch
import torch.nn as nn

from anatool import AnaLogger, AnaArgParser
from visdialch.paf.fc import MLP
from visdialch.paf.layer_norm import LayerNorm
from visdialch.paf.make_mask import make_mask
from visdialch.paf.model_cfgs import PAFCfgs
from visdialch.paf.paf_backbone import PAFBackbone


# flatten the sequence
class AttFlat(nn.Module):
    def __init__(self, paf_config: PAFCfgs, logger: AnaLogger):
        super(AttFlat, self).__init__()
        self.logger = logger
        self.paf_config = paf_config

        self.mlp = MLP(
            logger=logger,
            in_size=paf_config.HIDDEN_SIZE,
            mid_size=paf_config.FLAT_MLP_SIZE,
            out_size=paf_config.FLAT_GLIMPSES,
            dropout_r=paf_config.DROPOUT_R,
            use_relu=True
        )
        # FLAT_GLIMPSES == 1
        self.linear_merge = nn.Linear(
            in_features=paf_config.HIDDEN_SIZE * paf_config.FLAT_GLIMPSES,
            out_features=paf_config.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        # self.logger.debug(x.size())  # (bs*rounds, length, emb_size)
        att = self.mlp(x)
        # self.logger.debug(att.size())  # (bs*rounds, length, 1)
        if x_mask is not None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -1e9
            )
        att = nn.functional.softmax(att, dim=1)
        # self.logger.debug(att.size())  # (bs*rounds, length, 1)

        att_list = []
        for i in range(self.paf_config.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )
            #  MLP attention wise sum for each example --> note dim is 1 here

            # z = torch.sum(att[:, :, i: i + 1] * x, dim=1)
            # print(z.size())  # (bs*round, emb_size)

            x_atted = torch.cat(att_list, dim=1)
            # print(x_atted.size())  # (bs*round, emb_size)
            x_atted = self.linear_merge(x_atted)

            return x_atted


class PAF(nn.Module):
    def __init__(self, paf_config: PAFCfgs, logger: AnaLogger, answer_size, only_return_y=False):
        super(PAF, self).__init__()
        self.logger = logger
        self.paf_config = paf_config

        self.backbone = PAFBackbone(
            paf_config=paf_config,
            logger=logger
        )

        # flatten to vector
        self.attflat_img = AttFlat(paf_config=paf_config, logger=logger)
        self.attflat_lang = AttFlat(paf_config=paf_config, logger=logger)

        # classification layers
        self.proj_norm = LayerNorm(logger=logger, size=paf_config.FLAT_OUT_SIZE)
        self.proj = nn.Linear(in_features=paf_config.FLAT_OUT_SIZE, out_features=answer_size)
        self.only_return_y = only_return_y

    def forward(self, img_feat, lang_feat, lang_feat_mask, return_sep_modes=False):
        """
        :param img_feat: same embedding size as language (bs, num_boxes, 512/1024)
        :param lang_feat: (bs, max_length, 512/1024)
        :param lang_feat_mask:
        :param return_sep_modes:
        """
        img_feat_mask = make_mask(feature=img_feat)
        # self.logger.debug(img_feat_mask.size())

        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        if return_sep_modes:
            return lang_feat, img_feat

        # self.logger.debug(lang_feat.size())
        # flatten to vector
        lang_feat = self.attflat_lang(
            x=lang_feat,
            x_mask=lang_feat_mask
        )

        # self.logger.debug(lang_feat.size())  # (bs*round, 1024)

        img_feat = self.attflat_img(
            x=img_feat,
            x_mask=img_feat_mask
        )
        # self.logger.debug(img_feat.size())

        if self.only_return_y:
            return img_feat

        # element wise sum
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat


if __name__ == '__main__':
    parser = AnaArgParser()
    opt = parser.cfg
    logger = AnaLogger(opt.exp_dir)
    af = AttFlat(
        paf_config=PAFCfgs(),
        logger=logger
    )
    af(
        x=torch.rand(1280, 16, 512),
        x_mask=torch.ones(1280, 1, 1, 16)
    )
    paf = PAF(
        paf_config=PAFCfgs(),
        logger=logger,
        answer_size=512,
    )
    paf(
        img_feat=torch.rand(32, 36, 512),
        lang_feat=torch.rand(32, 20, 512),
        lang_feat_mask=torch.ones(32, 1, 1, 20)
    )
    logger.info(paf)
