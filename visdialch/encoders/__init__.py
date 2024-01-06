from anatool import AnaLogger, AnaArgParser
from visdialch.data.vocabulary import Vocabulary
from visdialch.encoders.cap_hist_early_fusion import CaptionHistoryEarlyFusionEncoder
# from visdialch.encoders.ic_hist_ques_fusion import ICHistoryQuestionFusionEncoder
# from visdialch.encoders.ic_hist_late_fusion import ICHistoryLateFusionEncoder
# from visdialch.encoders.ic_ques_hist_fusion import ICQuestionHistoryFusionEncoder
# from visdialch.encoders.iminic_ques_hist_fusion import IMiniCQuestionHistoryFusionEncoder


def encoders(opt, logger: AnaLogger, vocabulary):
    name_enc_map = {
        # 'ic_hist_ques_fusion': ICHistoryQuestionFusionEncoder,
        # 'ic_hist_late_fusion': ICHistoryLateFusionEncoder,
        'cap_hist_early_fusion': CaptionHistoryEarlyFusionEncoder,
        # 'ic_ques_hist_fusion': ICQuestionHistoryFusionEncoder,
        # 'iminic_ques_hist_fusion': IMiniCQuestionHistoryFusionEncoder,
    }
    return name_enc_map[opt['encoder']](
        opt=opt,
        logger=logger,
        vocabulary=vocabulary,
    )


if __name__ == '__main__':
    parser = AnaArgParser()
    opt = parser.cfg
    logger = AnaLogger(opt.exp_dir)
    name_enc_map = {
        # 'ic_hist_ques_fusion': ICHistoryQuestionFusionEncoder,
        # 'ic_hist_late_fusion': ICHistoryLateFusionEncoder,
        'cap_hist_early_fusion': CaptionHistoryEarlyFusionEncoder,
        # 'ic_ques_hist_fusion': ICQuestionHistoryFusionEncoder,
        # 'iminic_ques_hist_fusion': IMiniCQuestionHistoryFusionEncoder,
    }
    for k, v in name_enc_map.items():
        model = v(opt=opt, logger=logger, vocabulary=Vocabulary(
            word_counts_path=opt.word_counts_json,
            min_count=opt.vocab_min_count,
            logger=logger
        ))
        total = sum([param.nelement() for param in model.parameters()])
        logger.info(f"Number of {k} parameter: %.2fM" % (total / 1e6))
