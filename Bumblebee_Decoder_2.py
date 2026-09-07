import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadSelfAttention
from Bumblebee_Encoder import EncoderBlock

class DecoderBlock(nn.Module):
    def __init__(self, dropout_p, size_emb, n_heads, expansion):
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(size_emb, n_heads)
        
        #decoder is made out of a special attention block
        #with a diagonal mask and then the same as encoderblock
        self.EncoderBlock = EncoderBlock(dropout_p, size_emb, n_heads, expansion)

        #normalisation
        self.LayerNorm = nn.LayerNorm(size_emb)
        self.Dropout = nn.Dropout(dropout_p)

    def forward(self, input, K, V, pad_mask, diag_mask):
        attention = self.Dropout(self.attention(input, input, input, diag_mask))
        add = attention + input 
        q = self.LayerNorm(add)
        return self.EncoderBlock(q, K, V, pad_mask)




class Decoder(nn.Module):
    def __init__(self, device, vocab_size, size_embed, vocab_max, n_layers, dropout_p, n_heads, expansion):
        super().__init__()
        self.device = device
        self.layers = self.make_layers(n_layers, dropout_p, size_embed, n_heads, expansion)
        self.size_embed = size_embed
        self.dropout_p = dropout_p
        self.n_heads = n_heads
        self.expansion = expansion
        self.trg_vocab = vocab_size

        self.embedding = nn.Embedding(vocab_size, size_embed)
        self.pos_embedding = nn.Embedding(vocab_max, size_embed)

        self.Dropout = nn.Dropout(dropout_p)

        #fully connected output layer
        self.output_layer = nn.Linear(size_embed, vocab_size)

    def forward(self, input, output, pad_mask, diag_mask):
        n, input_len = input.shape
        #get positions of words in input to make embeddings. Add positional embeding to word embedding to get 
        #positional information into the embeddings
        input_positions = torch.arange(0, input_len).expand(n, input_len).to(self.device)
        a = self.embedding(input)
        b = self.pos_embedding(input_positions)
        embedd = a + b
        res = self.Dropout(embedd)

        #repeat the internal encoder loop n_layers times
        for layer in self.layers:
            res = layer(res, output, output, pad_mask, diag_mask)

        return self.output_layer(res)


    def make_layers(self, n_layers, dropout_p, size_embed, n_heads, expansion):
        return nn.ModuleList([DecoderBlock(dropout_p, size_embed, n_heads, expansion) for i in range(n_layers)])
