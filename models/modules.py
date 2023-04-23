import numpy as np
import torch
import torch.nn as nn
import unittest
import torch.nn.functional as F
from typing import Optional, Union, Tuple, Type, Any

from torch import FloatTensor, Tensor

# TODO : Make MultiHeadAttention
# TODO : Make Block
# TODO : Make Encoder
# TODO : Make Decoder
# TODO : Make LayerNorm


"""

B       :       batch size
S       :       source sequence length 
T       :       target sequence length
D       :       embedding dimension 


We will make a table of tensors dimension to make our life easier 

------------------------------------------
Tensor Name         | Tensor Dim
------------------------------------------
Source/Target (integer tensor) [B, S or T]
------------------------------------------
Embedded Source (float tensor) [B, S, D].            
------------------------------------------
Embedded source reshape  [B, S, H, D/H]
------------------------------------------
                    |                                    
"""


class MultiHeadAttention(nn.Module):

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int) -> None:

        super(MultiHeadAttention, self).__init__()

        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.qkv_dim = self.hidden_dim // self.num_heads
        self.qkv_projection = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.BoolTensor,
            cross_attention: bool = False,
    ):

        """
        We realise the forward pass of the MultiHeadAttentionBlock. We query the keys and values, and then compute the
        resulting attention score. We then project the values using the output projection.
        :param x: source sequence tensor. Shape : [B, S, E]
        :param mask: mask for the scaled dot product attention
        :param cross_attention: indicates whether we do cross attention.
        :return: output projection.Shape : [B, E, E]
        """

        B, S, E = x.size()
        if cross_attention:
            queries, keys, values = self.cross_attention_projection(x)
        else:
            queries, keys, values = self.self_attention_projection(x)

        queries, keys, values = map(
            lambda x: x.permute(0, 2, 1, 3),
            [queries, keys, values]
        )

        values, attn_score = self.scaled_dot_product_attention(
            queries=queries,
            keys=keys,
            values=values,
            mask=mask
        )

        values = values.reshape(B, S, E)
        output = self.output_projection(values)
        return output

    def self_attention_projection(
            self,
            x: torch.Tensor) -> Tuple[Any, Any, Any]:

        """
        An auxiliary method for the projection that takes care of reshaping various tensors. We pass in
        a source/target sequence of shape [B, S or T, E], reshapes it into [B, S, H, E//H] and outputs
        queries,  keys and values

        :param x: source sequence tensor of size [B, S or T, E]
        :return: queries, keys and values of shape [B, S, H, E/H]
        """

        B, S, E = x.shape
        qkv_proj = self.qkv_projection(x)
        qkv_proj = qkv_proj.view(B, S, self.num_heads, 3 * self.qkv_dim)
        queries, keys, values = torch.chunk(qkv_proj, chunks=3, dim=-1)
        return queries, keys, values

    def cross_attention_projection(
            self,
            x: torch.Tensor
    ) -> Tuple[Any, Any, Any]:
        raise NotImplementedError

    def cross_attention_projection(
            self,
            encoder_hidden_state: torch.Tensor,
            decoder_hidden_state: torch.Tensor
    ) -> Tuple[Type[FloatTensor], Tensor]:

        batch_size, src_sequence_length, embed_dim = encoder_hidden_state.shape
        batch_size, target_sequence_length, embed_dim = decoder_hidden_state.shape

    def scaled_dot_product_attention(
            self,
            queries: torch.FloatTensor,
            keys=torch.FloatTensor,
            values=torch.FloatTensor,
            source_padding_mask: Optional[torch.BoolTensor] = None,
            future_padding_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[Type[FloatTensor], Tensor]:
        """

        We implement the simple scaled dot product attention. We let

        B       : batch size
        H       : number of heads
        S       : source sequence length
        T       : target sequence length
        E       : embedding dimensionality


        :param queries: tensor containing queries. Shape : [B, H, S or T, E // H]
        :param keys: tensor containing the keys. Shape : [B, H, S or T, E // H]
        :param values: tensor containing the value tokens : Shape : [B, H, S or T, E // H]
        :param source_padding_mask: optional masking of source sequence.
        :param future_padding_mask: optional masking of future values
        :return: values, attention score. Shape of attention score : [B, S or T, E]
        """

        attention_keys = torch.matmul(queries, torch.transpose(keys, -2, -1))
        attention_keys /= torch.sqrt(self.queries.size()[-1])
        attention_logits = F.softmax(attention_keys, dim=-1)

        if source_padding_mask is not None or future_padding_mask is not None:
            self.mask_future_values(attention_logits, source_padding_mask, future_padding_mask)

        attention_score = torch.matmul(attention_logits, values)
        return values, attention_score

    @staticmethod
    def mask_future_values(
            logits: torch.Tensor,
            source_padding_mask: torch.BoolTensor,
            future_padding_mask: torch.BoolTensor
    ):

        global masked_logits
        if source_padding_mask is not None:
            masked_logits = torch.masked_fill(logits, source_padding_mask[:, None, None, :], np.NINF)
        if future_padding_mask is not None:
            masked_logits = torch.masked_fill(logits, future_padding_mask[:, None, None, :], np.NINF)
        return masked_logits
