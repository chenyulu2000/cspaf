import torch.nn as nn


class EncoderDecoderModel(nn.Module):
    """
    convenience wrapper module, wrapping encoder and decoder modules
    """

    def __init__(self, encoder, decoder):
        super(EncoderDecoderModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch, is_cfq=False, is_cfi=False):
        encoder_output = self.encoder(batch, is_cfq, is_cfi)
        decoder_output = self.decoder(encoder_output, batch)
        return decoder_output
