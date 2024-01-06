import torch.nn as nn

from anatool import AnaLogger


# full-connected layer
class FC(nn.Module):
    def __init__(self, logger: AnaLogger, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.logger = logger
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_features=in_size, out_features=out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


# multi-layer perception
class MLP(nn.Module):
    def __init__(self, logger: AnaLogger, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(
            logger=logger,
            in_size=in_size,
            out_size=mid_size,
            dropout_r=dropout_r,
            use_relu=use_relu
        )
        self.linear = nn.Linear(in_features=mid_size, out_features=out_size)

    def forward(self, x):
        return self.linear(self.fc(x))
