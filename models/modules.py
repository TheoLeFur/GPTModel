import torch
import torch.nn as nn
import numpy as np
import unittest
import math
import torch.nn.functional as F
from typing import Optional, Tuple
from torch import FloatTensor, Tensor


# TODO : Implement masking of future values in the decoder to preserve autoregressive property.

class MultiHeadAttention(nn.Module):

    def __init__(self,
                 block_size: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 ):

        """

        Multiple head attention module. We linearly project the queries, keys and values onto the learned projections,
        and then incorporate the attention mechanism in parallel. We then aggregate the attention scores of each head
        and pass it through an output projection layer.

        :param block_size: necessary for masking
        :param d_model: hidden dimension across the model, 512 by default
        :param n_heads: number of heads in which we run the attention mechanism in parallel.

        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.qkv_dim = self.d_model // self.n_heads
        self.block_size = block_size

        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(
            self.block_size, self.block_size
        )).view(1, 1, self.block_size, self.block_size)
                             )

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
            mask=self.mask
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
        T = queries.shape[2]
        attention_logits = torch.matmul(queries, torch.transpose(keys, -1, -2)) / math.sqrt(self.qkv_dim)
        if mask is not None:
            torch.masked_fill(attention_logits, mask[:, :, :T, :T] == 0, np.NINF)
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
            LayerNorm(self.d_model),
            MultiHeadAttention(self.n_heads),
        )

        self.feedforward_nn_block = nn.Sequential(
            LayerNorm(self.d_model),
            DoubleMLP(
                config=self.config,
                hidden_activation=self.hidden_activation,
                output_activation=self.output_activation
            )
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


class TransformerDecoder(nn.Module):

    def __init__(self,
                 config,
                 ) -> None:
        """
        We create a transformer decoder, which will be the only one we need for a generative pretrained transformer.
        Hence, we do not make usage of the full transformer architecture.

        :param config: configure the params for the model
        """

        super(TransformerDecoder, self).__init__()

        self.n_blocks = config["n_blocks"]
        self.vocab_len = config["vocab_len"]
        self.max_len = config["max_len"]
        self.d_model = config["d_model"]
        self.dropout_p = config["dropout"]
        self.bias = config["bias"]
        self.word_embedding = nn.Embedding(self.vocab_len, self.d_model)
        self.pos_encoding = PositionalEncoding(config)
        self.dropout = nn.Dropout(self.dropout_p)

        self.dropout = nn.Dropout(self.dropout_p)
        module_list = [TransformerBlock(config) for _ in range(self.n_blocks)]
        self.transformer_decoder = nn.Sequential(
            *module_list,
            LayerNorm(self.d_model)
        )
        self.head = nn.Linear(self.d_model, self.vocab_len, bias=self.bias)

    def forward(self,
                idx: torch.Tensor,
                targets: torch.Tensor = None) -> Tuple[torch.Tensor, Optional[Tensor]]:
        b, t = idx.size()
        x = self.dropout(self.word_embedding(idx) + self.pos_encoding(idx))
        x = self.transformer_decoder(x)

        logits = self.head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        else:
            loss = None

        return logits, loss

if __name__ == "__main__":

    config = dict(
        vocab_len = 10000,
        max_len = 100,
        d_model = 512,
        dropout = 0.1,
        bias = False,
        n_blocks = 12,
        device = "cpu",
        n_heads = 8
    )

    idx = torch.randn(32, 100)
    decoder = TransformerDecoder(config)
    decoder(idx)

