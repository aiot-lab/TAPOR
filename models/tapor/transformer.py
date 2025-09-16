import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
# from mmcv.cnn import xavier_init
from torch import Tensor
import math

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class SinePositionalEncoding(nn.Module):
    """Position encoding with sine and cosine functions.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.,
                 init_cfg=None):
        super(SinePositionalEncoding, self).__init__()
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                                                    'scale should be provided and in float or int type, ' \
                                                    f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask):
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # [bs, h, w], recording the y coordinate ot each pixel
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:  # default True
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t  # [bs, h, w, num_feats]
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, H, W = mask.size()
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)  # [bs, h, w, num_feats]
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class LearnedPositionalEncoding(nn.Module):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super(LearnedPositionalEncoding, self).__init__(init_cfg)
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat(
            (x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(
                1, w, 1)),
            dim=-1).permute(2, 0,
                            1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'
        return repr_str


def position_embedding(seq_len, d_model):

    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe

class HandposeEncoder(nn.Module):
    def __init__(self, d_model, kv_dim,h,w,batch,c,nhead, dim_feedforward=2048, dropout=0.1,num_layers = 3, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(HandposeEncoder,self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.position = SinePositionalEncoding(c//2)
        self.position_embedding = torch.zeros(batch, h,w).to(device)
        self.position_embedding = self.position(self.position_embedding)
        # print(self.position_embedding.shape)
        self.att1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout,kdim=kv_dim,vdim=kv_dim)

        self.d_model = d_model
        self.kv_dim = kv_dim
        self.encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(self.encoder_layer, num_layers, self.encoder_norm)
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src, support_embed):
        bs, c, h, w = src.shape
        src =self.with_pos_embed(src, self.position_embedding)
        src = src.flatten(2).permute(2, 0, 1)  # [hw, bs, c]
        query_embed = support_embed.transpose(0, 1)
        # pe_q = position_embedding(support_embed.shape[1], self.d_model).unsqueeze(1).repeat(1, bs, 1).to(self.device)
        # pe_kv = position_embedding(src.shape[0], self.kv_dim).unsqueeze(1).repeat(1, bs, 1).to(self.device)
        # query_embed = self.with_pos_embed(query_embed, pe_q)
        # src = self.with_pos_embed(src, pe_kv)    
        output, qk_weights = self.att1(query_embed, src, src)
        # print(  output)
        # pos_embed = position_embedding(output.shape[0], self.d_model).unsqueeze(1).repeat(1, bs, 1).to(self.device)
        pos_embed = None
        query_embed = self.encoder(output,pos=pos_embed)
        return query_embed, output, qk_weights

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                src,
                pos: Optional[Tensor] = None):
        # src: [hw, bs, c]
        # query: [num_query, bs, c]
        # mask: None by default
        # src_key_padding_mask: [bs, hw]
        # query_key_padding_mask: [bs, nq]
        # pos: [hw, bs, c]

        # organize the input
        # implement the attention mask to mask out the useless points
        # n, bs, c = src.shape
        # src_cat = torch.cat((src, query), dim=0)  # [hw + nq, bs, c]
        # mask_cat = torch.cat((src_key_padding_mask, query_key_padding_mask),
        #                      dim=1)  # [bs, hw+nq]
        output = src

        for layer in self.layers:
            output = layer(
                output,
                pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        ## resplit the output into src and query
        # refined_query = output[n:, :, :]  # [nq, bs, c]
        # output = output[:n, :, :]  # [n, bs, c]

        return output

class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                src,
                pos: Optional[Tensor] = None):
        src = self.with_pos_embed(src, pos)
        q = k = src
        # NOTE: compared with original implementation, we add positional embedding into the VALUE.
        src2 = self.self_attn(
            q,
            k,
            value=src,
            )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    en = HandposeEncoder(256, 32*24,32,24,10,24,8).to(device)
    src = torch.rand(10, 24, 32, 24).to(device)
    # mask = torch.ones(10, 96, 72).bool()
    support_embed = torch.rand(10, 21, 256).to(device)
    # masks = src.new_zeros((src.shape[0], src.shape[2], src.shape[3])).to(torch.bool)
    # position = SinePositionalEncoding(22)
    # pos_embed = position(masks)
    # print(pos_embed.shape)
    # # pos_embed = None
    # # pos_embed = torch.rand(10, 256, 21)
    # support_order_embed = torch.rand(10, 256, 21)
    # query_padding_mask = torch.ones(10, 21).bool()
    # # position_embedding = torch.rand(10, 256, 21)
    # # kpt_branch = torch.rand(10, 256, 21)
    # # skeleton = torch.rand(10, 256, 21)
    query_embed = en(src, support_embed)
    print(query_embed.shape)
    # print(refined_support_embed.shape)

