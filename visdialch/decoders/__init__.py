from anatool import AnaLogger
from visdialch.decoders.disc import DiscriminativeDecoder


def decoders(opt, logger: AnaLogger, vocabulary):
    name_dec_map = {
        'disc': DiscriminativeDecoder,
    }
    return name_dec_map[opt['decoder']](
        opt=opt,
        logger=logger,
        vocabulary=vocabulary,
    )
