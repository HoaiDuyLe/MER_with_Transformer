import math, copy
from typing import Optional, List
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from src.utils import padTensor
from timm.models.vision_transformer import DropPath, Mlp, Attention

class BuildTransformerDecoder(nn.Module):
    def __init__(self, dim, num_layers, num_heads):
        super(BuildTransformerDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
    
    def forward(self, query_embed, memory, memory_key_padding_mask):
        query_embed = query_embed.unsqueeze(1).repeat(1, memory.shape[1], 1)
        output = self.decoder(query_embed, memory, memory_key_padding_mask=memory_key_padding_mask)
        return output

class WrappedTransformerEncoder(nn.Module):
    def __init__(self, dim, num_layers, num_heads):
        super(WrappedTransformerEncoder, self).__init__()
        self.dim = dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=dim)
        self.pos_encoder = PositionalEncoding(dim, dropout=0.1)

    def prepend_cls(self, inputs):        
        cls_emb = self.cls_emb.weight
        cls_emb = cls_emb.unsqueeze(1).repeat(inputs.shape[0], 1, 1)
        outputs = torch.cat((cls_emb, inputs), dim=1)
        return outputs

    def forward(self, inputs: torch.Tensor, lens: Optional[List[int]] = None, get_cls: Optional[bool] = True, pos_en: Optional[bool] = True):
        if lens is not None:
            max_len = max(lens)

            mask = [([False] * (l + int(get_cls)) + [True] * (max_len - l)) for l in lens]
            mask = torch.tensor(mask).to(device=inputs.device)

            inputs = list(inputs.split(lens, dim=0))
            inputs = [padTensor(inp, max_len) for inp in inputs]
            inputs = torch.stack(inputs, dim=0)
        else:
            mask = None

        if get_cls:
            inputs = self.prepend_cls(inputs)

        inputs = inputs.permute(1, 0, 2) #(seq_len, bs, dim)
        if pos_en:
            inputs = self.pos_encoder(inputs) #(seq_len, bs, dim)
        inputs = self.encoder(src=inputs, src_key_padding_mask=mask) #(seq_len, bs, dim)
        if get_cls:
            return inputs[0], inputs[1:].permute(1, 0, 2), mask[:,1:]
            # return inputs[0], inputs.permute(1, 0, 2), mask
        else:
            return inputs[0], inputs.permute(1, 0, 2), mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


