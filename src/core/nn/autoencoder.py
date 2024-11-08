import torch
from torch import nn
import warnings

def _enc_layer_to_dec(layer : nn.Module):
    cls = layer.__class__
    if 'Lazy' in cls.__name__:
        raise NotImplementedError("Lazy layers are not currently supported")
    if cls.__module__ == nn.Conv2d.__module__ and 'Conv' in cls.__name__:
        inv_cls_name : str
        if 'Transpose' in cls.__name__:
            inv_cls_name = cls.__name__.replace('Transpose', '')
        else:
            inv_cls_name = cls.__name__.replace('Conv', 'ConvTranspose')
        inv_cls = getattr(nn, inv_cls_name)
        return inv_cls(
            layer.out_channels, 
            layer.in_channels, 
            layer.kernel_size, 
            layer.stride, 
            layer.padding
        )
    elif cls == nn.Linear:
        return nn.Linear(layer.out_features, layer.in_features)
    else:
        return cls()

class AutoEncoder(nn.Module):
    """AutoEncoder module

    """
    def __init__(self, encoder : nn.Module, decoder : nn.Module):
        """Creates an autoencoder with the given encoder and decoder

        Args:
            encoder (nn.Module): the encoder module
            decoder (nn.Module): the decoder module
        """
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    @classmethod
    def from_encoder(cls, encoder : nn.Module) -> 'AutoEncoder':
        """Attempts to automatically create the decoder from the encoder

        Args:
            encoder (nn.Module): The encoder model to use

        Returns:
            AutoEncoder: an autoencoder with the given encoder and a decoder that mirrors the encoder
        """
        warnings.warn('This method is experimental and may be removed in the future.')
        layers = list(encoder.children())
        decoder_layers = [_enc_layer_to_dec(layer) for layer in reversed(layers)]
        decoder = nn.Sequential(*decoder_layers)
        return cls(encoder, decoder)
