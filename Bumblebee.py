import torch
from torch import nn
from Bumblebee_Encoder import Encoder
from Bumblebee_Decoder_2 import Decoder

class Bumblebee(nn.Module):
    def __init__(self, device, 
                    target_vocab, 
                    src_vocab, 
                    size_embed, 
                    vocab_max, 
                    n_layers, 
                    dropout_p,
                    n_heads,
                    expansion, 
                    pad_idx):
        super(Bumblebee, self).__init__()

        self.decodeBlock = Decoder(device, target_vocab, size_embed, vocab_max, n_layers, dropout_p, n_heads, expansion)
        self.encodeBlock = Encoder(device, n_layers, dropout_p, size_embed, n_heads, expansion, src_vocab, vocab_max)
        self.device = device
        
        self.pad_idx = pad_idx

    def forward(self, source_inputs, target_inputs):
        decoder_mask = self.get_decoder_mask(target_inputs, self.device)
        encoder_mask = self.get_encoder_mask(source_inputs, self.pad_idx, self.device)
        source_encoding = self.encodeBlock(source_inputs, encoder_mask)
        return self.decodeBlock(target_inputs, source_encoding, encoder_mask, decoder_mask)

    def get_decoder_mask(self, input, device):
        #print(input.shape)
        n, len = input.shape
        return torch.tril(torch.ones((len, len))).expand(n, 1, len, len).to(device)

    def get_encoder_mask(self, src, pad_index, device):
        src_mask = (src != pad_index).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)

        return src_mask.to(device)