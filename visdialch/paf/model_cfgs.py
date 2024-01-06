class PAFCfgs:
    def __init__(self):
        self.LAYER = 6
        self.MULTI_HEAD = 8
        self.HIDDEN_SIZE = 512
        self.BBOX_FEAT_EMB_SIZE = 2048
        self.FF_SIZE = 2048
        self.DROPOUT_R = 0.1
        self.FLAT_MLP_SIZE = 512
        self.FLAT_GLIMPSES = 1
        self.FLAT_OUT_SIZE = 1024
        self.USING_PAF = True
