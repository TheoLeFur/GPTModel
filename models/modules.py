import numpy as np
import torch
import torch.nn as nn
import unittest
import math
import torch.nn.functional as F
from typing import Optional, Union, Tuple, Type, Any

from torch import FloatTensor, Tensor


# TODO : Make Block
# TODO : Make Encoder
# TODO : Make Decoder
# TODO : Make LayerNorm

class MultiHeadAttention(nn.Module):

    def __init__(self,
                 d_model: int = 512,
                 n_heads: int = 8):
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.qkv_dim = self.d_model // self.n_heads

        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.size()
        queries, keys, values = self.qkv_proj(x).split(self.d_model, dim=-1)

        # One now needs to reshape the tensors in order to compute the scaled dot product attention
        queries, keys, values = map(
            lambda x: x.view(B, T, self.n_heads, self.qkv_dim).transpose(1, 2),
            [queries, keys, values]
        )

        values, attn_score = self.scaled_dot_product_attention(
            queries,
            keys,
            values,
        )

        attn_score = attn_score.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(attn_score)

    def scaled_dot_product_attention(self,
                                     queries: torch.FloatTensor,
                                     keys: torch.FloatTensor,
                                     values: torch.FloatTensor,
                                     mask: Optional[torch.BoolTensor] = None) -> Tuple[FloatTensor, Tensor]:
        """
        Scaled dot product attention as presented in the Attention is All You Need Paper.

        :param queries: input shape of [B, H, T, D/H]
        :param keys: input shape of [B, H, T, D/H]
        :param values: input shape of [B, H, T, D/H]
        :param mask: optional masking out of illegal connections in the decoder's self attention mechanism. Prevents the corruption
        of the autoregressive property of the transformer.
        :return: value and attention score.
        """

        attention_logits = torch.matmul(queries, torch.transpose(keys, -1, -2)) / math.sqrt(self.qkv_dim)
        if mask is not None:
            torch.masked_fill(attention_logits, mask[:, None, None, :] == 0, np.NINF)
        softmax = F.softmax(attention_logits, dim=-1)
        # We multiply [B, H, T, T] * [B, H, T, D/H] -> [B, H, T, D/H]
        attn_score = torch.matmul(softmax, values)
        return values, attn_score


class LayerNorm(nn.Module):
    def __init__(self,
                 d_model: int = 512) -> None:
        """
        We implement Layer Normalisation, since the devs where too lazy to implement
        it for the mps gpu, one does not need to do it if you use cuda or your cpu.
        :param d_model: hidden dim of your model

        """

        super(LayerNorm, self).__init__()

        self.d_model = d_model
        self.gamma = nn.Parameter(torch.ones(self.d_model))
        self.beta = nn.Parameter(torch.zeros(self.d_model))
        self.eps = 1e-12

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer norm
        :param x: input tensor
        :return: learn normalization of the layer.
        """
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        out = (x - mean) / (torch.sqrt(var) + self.eps) * self.gamma + self.beta
        return out


class DoubleMLP(nn.Module):

    def __init__(self,
                 config: dict,
                 hidden_activation: nn.Module = None,
                 output_activation: nn.Module = None):

        super(DoubleMLP, self).__init__()

        self.config = config
        self.d_model = config["d_model"]
        self.dropout = config["dropout"]
        self.bias = config['bias']

        if hidden_activation is None:
            self.hidden_activation = nn.GELU()
        else:
            self.hidden_activation = hidden_activation
        if output_activation is None:
            self.output_activation = nn.Identity()
        else:
            self.output_activation = output_activation

        self.double_mlp = nn.Sequential(
            nn.Linear(self.d_model, 4 * self.d_model, bias=self.bias),
            self.hidden_activation,
            nn.Linear(4 * self.d_model, self.d_model, bias=self.bias),
            nn.Dropout(self.dropout),
            self.output_activation
        )

    def forward(self,
                x: torch.Tensor):

        """
        Forward pass for the double-headed MLP
        :param x: input tensor
        :return : output tensor
        """
        output = self.double_mlp(x)
        return output


class TransformerBlock(nn.Module):

    def __init__(self,
                 config: dict,
                 hidden_activation: nn.Module = None,
                 output_activation: nn.Module = None) -> torch.Tensor:

        super(TransformerBlock, self).__init__()

        self.config = config
        self.d_model = self.config["d_model"]
        self.bias = self.config["bias"]
        self.n_heads = config["n_heads"]

        if hidden_activation is None:
            self.hidden_activation = nn.GELU()
        else:
            self.hidden_activation = hidden_activation
        if output_activation is None:
            self.output_activation = nn.Identity()
        else:
            self.output_activation = output_activation

        self.attention_block = nn.Sequential(
            MultiHeadAttention(self.n_heads),
            LayerNorm(self.d_model)
        )

        self.feedforward_nn_block = nn.Sequential(
            DoubleMLP(
                config=self.config,
                hidden_activation=self.hidden_activation,
                output_activation=self.output_activation
            ),
            LayerNorm(self.d_model),
        )

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention_block(x)
        x = x + self.feedforward_nn_block(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self,
                 config: dict) -> None:
        super(PositionalEncoding, self).__init__()

        self.d_model = config["d_model"]
        self.device = config["device"]
        self.max_len = config["max_len"]

        self.encoding = torch.zeros(self.max_len, self.d_model, device=self.device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, self.max_len, device=self.device).float().unsqueeze(1)

        powers = torch.arange(0, self.d_model, step=2, device=self.device).float()

        self.encoding[:, 0::2] = torch.sin(pos /
                                           (10000 ** (powers / self.d_model)))
        self.encoding[:, 1::2] = torch.cos(pos /
                                           (10000 ** (powers / self.d_model)))

    def forward(self,
                x: torch.Tensor) -> torch.FloatTensor:
        """
        We return a positional encoding for the input token. we have to track somehow the position of the token in the
        sentence, else will we use order in the reconstruction. If we do not use this, phrase like :
        1. The cat ate the mouse
        2. The mouse ate the cat
        3. The the mouse ate cat

        Would be equally likely, which is something we do not want, obviously.

        :param x: tensor
        :returns positional encoding of the tensor
        """
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]


class TransformerEncoderBlock(nn.Module):

    def __init__(self,
                 ):