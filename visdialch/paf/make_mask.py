import torch

from anatool import AnaArgParser, AnaLogger


# making the sequence mask
def make_mask(feature):
    """
    :param feature:
        for img: (bs, proposals, 2048/512)
        for text - do text.unsqueeze(2) first : (bs, seq_len, 1)
    :return:
        shape: (bs, 1, 1, seq_len/proposal)
    """
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)


if __name__ == '__main__':
    parser = AnaArgParser()
    opt = parser.cfg
    logger = AnaLogger(opt.exp_dir)
    feature = torch.rand(1, 36, 1024)
    logger.debug(make_mask(feature).size())  # (1, 1, 1, 36)
