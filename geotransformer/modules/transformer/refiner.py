from pathlib import Path
from types import SimpleNamespace
import warnings
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, List, Callable
from time import perf_counter
import open3d as o3d
from easydict import EasyDict as edict

@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints(
        kpts: torch.Tensor,
        size: torch.Tensor) -> torch.Tensor:
    if not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
    return kpts


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)



class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """ get confidence tokens """
        return (
            self.token(desc0.detach().float()).squeeze(-1),
            self.token(desc1.detach().float()).squeeze(-1))


class Attention(nn.Module):
    def __init__(self, hist_bins=None) -> None:
        super().__init__()
        self.hist_bins = hist_bins

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        scaled_dot_product_attention을 직접 구현 (PyTorch 1.x 호환)
        """
        # q, k, v: (..., seq_len, dim)
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)  # scaled dot product

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)


class Transformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3*embed_dim, bias=bias)
        self.inner_attn = Attention()
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2*embed_dim, 2*embed_dim),
            nn.LayerNorm(2*embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2*embed_dim, embed_dim)
        )

    def _forward(self, x: torch.Tensor,
                 mask = None):
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        context = self.inner_attn(q, k, v, mask = mask)
        message = self.out_proj(
            context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))

    def forward(self, x0, x1, mask0=None, mask1=None):
        return self._forward(x0, mask = mask0), self._forward(x1, mask = mask1)


class CrossTransformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2*embed_dim, 2*embed_dim),
            nn.LayerNorm(2*embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2*embed_dim, embed_dim)
        )
        self.attention = Attention()

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> List[torch.Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
            (qk0, qk1, v0, v1))
        m0 = self.attention(qk0, qk1, v1)
        m1 = self.attention(qk1, qk0, v0)
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2),
                           m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


def log_double_softmax(
        sim: torch.Tensor) -> torch.Tensor:
    """ create the log assignment matrix from logits and similarity"""
    b, m, n = sim.shape
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim, 1)
    scores = scores0 + scores1
    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim: int, normalize_descriptors = False) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)
        self.normalize_descriptors = normalize_descriptors

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """ build assignment matrix from descriptors """
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        if self.normalize_descriptors:
            inv_temp = 20
            mdesc0, mdesc1 = inv_temp * mdesc0/mdesc0.norm(dim=-1,keepdim = True), inv_temp * mdesc1/mdesc1.norm(dim=-1,keepdim = True)
        else:
            mdesc0, mdesc1 = mdesc0 / d**.25, mdesc1 / d**.25

        sim = torch.einsum('bmd,bnd->bmn', mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = log_double_softmax(sim)
        return scores, z0, z1

class Refiner(nn.Module):
    def __init__(self, conf, **kwargs) -> None:
        super().__init__(**kwargs)
        h, n, d = conf.refiner_num_heads, conf.refiner_n_layers, conf.refiner_descriptor_dim
        self.num_heads = h
        self.self_attn = nn.ModuleList(
            [Transformer(d, h, conf.flash) for _ in range(n)])
        self.cross_attn = nn.ModuleList(
            [CrossTransformer(d, h, conf.flash) for _ in range(n)])
        self.conf = conf
    
    def forward(self, desc0, desc1):
        mask0, mask1 = None, None

        for i in range(self.conf.refiner_n_layers):
            desc0, desc1 = self.self_attn[i](desc0, desc1, mask0 = mask0, mask1 = mask1)
            desc0, desc1 = self.cross_attn[i](desc0, desc1)
        return desc0, desc1
