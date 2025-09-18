import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import square_dists, gather_points, sample_and_group, angle
from geotransformer.modules.transformer.rpe_transformer import RPETransformerLayer
from geotransformer.modules.transformer.vanilla_transformer import TransformerLayer


def get_graph_features(feats, coords, k=10):
    '''

    :param feats: (B, N, C)
    :param coords: (B, N, 3)
    :param k: float
    :return: (B, N, k, 2C)
    '''

    sq_dists = square_dists(coords, coords)
    n = coords.size(1)
    inds = torch.topk(sq_dists, min(n, k+1), dim=-1, largest=False, sorted=True)[1]
    inds = inds[:, :, 1:] # (B, N, k)

    neigh_feats = gather_points(feats, inds) # (B, N, k, c)
    # feats = torch.unsqueeze(feats, 2).repeat(1, 1, min(n-1, k), 1) # (B, N, k, c)
    feats = feats.unsqueeze(2).repeat(1, 1, min(n-1, k), 1) # (B, N, k, c)
    
    return torch.cat([feats, neigh_feats - feats], dim=-1)


class LocalFeatureFused(nn.Module):
    def __init__(self, in_dim, out_dims):
        super(LocalFeatureFused, self).__init__()
        self.blocks = nn.Sequential()
        for i, out_dim in enumerate(out_dims):
            self.blocks.add_module(f'conv2d_{i}',
                                   nn.Conv2d(in_dim, out_dim, 1, bias=False))
            self.blocks.add_module(f'in_{i}',
                                   nn.InstanceNorm2d(out_dims))
            self.blocks.add_module(f'relu_{i}', nn.ReLU(inplace=True))
            in_dim = out_dim

    def forward(self, x):
        '''
        :param x: (B, C1, K, M)
        :return: (B, C2, M)
        '''
        x = self.blocks(x)
        x = torch.max(x, dim=2)[0]
        return x

class PPF(nn.Module):
    def __init__(self, feats_dim, k, radius):
        super().__init__()
        self.k = k
        self.radius = radius
        self.local_feature_fused = LocalFeatureFused(in_dim=10,
                                                     out_dims=feats_dim)
    def forward(self, coords: torch.Tensor, normals: torch.Tensor):
        """
        Args:
            coords: (B, 3, N)
            normals: (B, 3, N)
        Returns:
            feats_ppf: (B, C, N)
        """
        B, _, N = coords.shape
        coords = coords.permute(0, 2, 1).contiguous()   # (B, N, 3)
        normals = normals.permute(0, 2, 1).contiguous() # (B, N, 3)

        # PPF 기반 local group 추출
        _, grouped_feats, grouped_inds, grouped_xyz = sample_and_group(
            xyz=coords,
            points=normals,
            M=-1,
            radius=self.radius,
            K=self.k,
            use_xyz=True,
            rt_density=False
        )  # grouped_xyz: (B, N, K, 3), grouped_feats: (B, N, K, C)
        
        grouped_xyz = grouped_xyz.to(normals.device)
        
        
        # PPF 생성
        # feats: normals → (B, N, C), grouped_xyz: (B, N, K, 3)
        B, N, K, _ = grouped_xyz.shape
        normals_expanded = normals.unsqueeze(2).expand(-1, -1, K, -1)  # (B, N, K, 3)
        ppf_angle = angle(normals_expanded, grouped_xyz)  # (B, N, K)

        # pooling: max across K neighbors
        ppf_angle = ppf_angle.max(dim=2)[0]  # (B, N)
        # 임시로 하나의 feature map만 사용 (추후 concat or FC 가능)
        feats_ppf = ppf_angle.unsqueeze(1)  # (B, 1, N)
  
        return feats_ppf

class GCN(nn.Module):
    def __init__(self, feats_dim, k):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(feats_dim * 2, feats_dim, 1, bias=False),
            nn.InstanceNorm2d(feats_dim),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(feats_dim * 2, feats_dim * 2, 1, bias=False),
            nn.InstanceNorm2d(feats_dim * 2),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(feats_dim * 4, feats_dim, 1, bias=False),
            nn.InstanceNorm1d(feats_dim),
            nn.LeakyReLU(0.2)
        )
        self.k = k

    def forward(self, coords, feats):
        '''

        :param coors: (B, 3, N)
        :param feats: (B, C, N)
        :param k: int
        :return: (B, C, N)
        '''

        feats1 = get_graph_features(feats=feats.permute(0, 2, 1).contiguous(),
                                    coords=coords.permute(0, 2, 1).contiguous(),
                                    k=self.k)
        feats1 = self.conv1(feats1.permute(0, 3, 1, 2).contiguous())
        feats1 = torch.max(feats1, dim=-1)[0]

        feats2 = get_graph_features(feats=feats1.permute(0, 2, 1).contiguous(),
                                    coords=coords.permute(0, 2, 1).contiguous(),
                                    k=self.k)
        feats2 = self.conv2(feats2.permute(0, 3, 1, 2).contiguous())
        feats2 = torch.max(feats2, dim=-1)[0]

        feats3 = torch.cat([feats, feats1, feats2], dim=1)
        feats3 = self.conv3(feats3)

        return feats3


class GGE(nn.Module):
    def __init__(self, feats_dim, gcn_k, ppf_k, radius, bottleneck):
        super().__init__()
        self.gcn = GCN(feats_dim, gcn_k)
        if bottleneck:
            self.ppf = PPF([feats_dim // 2, feats_dim, feats_dim // 2], ppf_k, radius)
            self.fused = nn.Sequential(
                nn.Conv1d(384, 512, 1),
                nn.InstanceNorm1d(feats_dim + feats_dim // 2),
                nn.LeakyReLU(0.2),
                nn.Conv1d(feats_dim + feats_dim // 2, feats_dim, 1),
                nn.InstanceNorm1d(feats_dim),
                nn.LeakyReLU(0.2)
                )
        else:
            self.ppf = PPF([feats_dim, feats_dim*2, feats_dim], ppf_k, radius)
            self.fused = nn.Sequential(
                nn.Conv1d(257, 512, 1),
                nn.InstanceNorm1d(feats_dim * 2),
                nn.LeakyReLU(0.2),
                nn.Conv1d(feats_dim * 2, feats_dim, 1),
                nn.InstanceNorm1d(feats_dim),
                nn.LeakyReLU(0.2)
                )
    
    def forward(self, coords, feats, normals):
        feats_ppf = self.ppf(coords, normals)
        feats_gcn = self.gcn(coords, feats)
        # feats_ppf = self.high_ppf(coords, normals)
        # feats_gcn = self.delited_gcn(coords, feats)
    
        feats_ppf = feats_ppf.to(feats_gcn.device)
        feats_fused = self.fused(torch.cat([feats_ppf, feats_gcn], dim=1))

        return feats_fused

class AugmentedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, shape=0, relative=False, stride=1):
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divisible by Nh"
        assert self.dv % self.Nh == 0, "dv should be divisible by Nh"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        # conv branch
        self.conv_out = nn.Conv2d(
            self.in_channels,
            self.out_channels - self.dv,
            kernel_size=(1, self.kernel_size),
            stride=(1, stride),
            padding=(0, self.padding)
        )

        # QKV projection
        self.qkv_conv = nn.Conv2d(
            self.in_channels,
            2 * self.dk + self.dv,
            kernel_size=(1, self.kernel_size),
            stride=(1, stride),
            padding=(0, self.padding)
        )

        # output projection
        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

    def forward(self, x):
        # x: (B, C, N)
        x = x.unsqueeze(2)  # (B, C, 1, N)

        conv_out = self.conv_out(x)  # (B, out_channels-dv, 1, N')
        batch, _, _, width = conv_out.size()

        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        weights = F.softmax(logits, dim=-1)

        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, 1, width))
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)

        out = torch.cat((conv_out, attn_out), dim=1)  # (B, out_channels, 1, N')
        return out.squeeze(2)  # (B, out_channels, N')

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        N, _, _, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q = q * (dkh ** -0.5)  # <-- out-of-place로 변경
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, _, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, 1, width)
        return torch.reshape(x, ret_shape)

    def combine_heads_2d(self, x):
        batch, Nh, dv, _, width = x.size()
        return torch.reshape(x, (batch, Nh * dv, 1, width))


def multi_head_attention(query, key, value):
    '''
    :param query: (B, dim, nhead, N)
    :param key: (B, dim, nhead, M)
    :param value: (B, dim, nhead, M)
    :return: (B, dim, nhead, N)
    '''
    dim = query.size(1)
    scores = torch.einsum('bdhn, bdhm->bhnm', query, key) / dim**0.5
    attention = torch.nn.functional.softmax(scores, dim=-1)
    feats = torch.einsum('bhnm, bdhm->bdhn', attention, value)
    return feats


class Cross_Attention(nn.Module):
    def __init__(self, feat_dims, nhead):
        super().__init__()
        assert feat_dims % nhead == 0
        self.feats_dim = feat_dims
        self.nhead = nhead
        # self.q_conv = nn.Conv1d(feat_dims, feat_dims, 1, bias=True)
        # self.k_conv = nn.Conv1d(feat_dims, feat_dims, 1, bias=True)
        # self.v_conv = nn.Conv1d(feat_dims, feat_dims, 1, bias=True)
        self.conv = nn.Conv1d(feat_dims, feat_dims, 1)
        self.q_conv, self.k_conv, self.v_conv = [copy.deepcopy(self.conv) for _ in range(3)] # a good way than better ?
        self.mlp = nn.Sequential(
            nn.Conv1d(feat_dims * 2, feat_dims * 2, 1),
            nn.InstanceNorm1d(feat_dims * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(feat_dims * 2, feat_dims, 1),
        )

    def forward(self, feats1, feats2):
        '''
        :param feats1: (B, C, N)
        :param feats2: (B, C, M)
        :return: (B, C, N)
        '''         
        b = feats1.size(0)
        dims = self.feats_dim // self.nhead
        query = self.q_conv(feats1).reshape(b, dims, self.nhead, -1)
        key = self.k_conv(feats2).reshape(b, dims, self.nhead, -1)
        value = self.v_conv(feats2).reshape(b, dims, self.nhead, -1)
        feats = multi_head_attention(query, key, value)
        feats = feats.reshape(b, self.feats_dim, -1)
        feats = self.conv(feats)
        cross_feats = self.mlp(torch.cat([feats1, feats], dim=1))
        
        return cross_feats


class InformationInteractive(nn.Module):
    def __init__(self,
                layer_names, 
                feat_dims, 
                gcn_k, 
                ppf_k, 
                radius, 
                bottleneck, 
                nhead):
        super().__init__()
        self.layer_names = layer_names
        self.blocks = nn.ModuleList()
        for layer_name in layer_names:
            if layer_name == 'gcn':
                self.blocks.append(GCN(feat_dims, gcn_k))
            elif layer_name == 'gge':
                self.blocks.append(GGE(feat_dims, gcn_k, ppf_k, radius, bottleneck))
            elif layer_name == 'cross_attn':
                self.blocks.append(Cross_Attention(feat_dims, nhead))
            elif layer_name == 'self':
                self.blocks.append(AugmentedConv(in_channels=feat_dims,
                                                 out_channels=feat_dims,
                                                 kernel_size=3,
                                                 dk=64,
                                                 dv=64,
                                                 Nh=nhead,
                                                 shape=0,
                                                 relative=False,
                                                 stride=1))
            else:
                raise NotImplementedError

    def forward(self, coords1, feats1, coords2, feats2, normals1, normals2):
        '''
        :param coords1: (B, 3, N)
        :param feats1: (B, C, N)
        :param coords2: (B, 3, M)
        :param feats2: (B, C, M)
        :return: feats1=(B, C, N), feats2=(B, C, M)
        '''
        
        for layer_name, block in zip(self.layer_names, self.blocks):
            if layer_name == 'gcn':
                feats1 = block(coords1, feats1)
                feats2 = block(coords2, feats2)
            elif layer_name == 'gge':
                feats1 = block(coords1, feats1, normals1)
                feats2 = block(coords2, feats2, normals2)
            elif layer_name == 'cross_attn':
                feats1 = feats1 + block(feats1, feats2)
                feats2 = feats2 + block(feats2, feats1)
            elif layer_name == 'self':
                feats1 = block(feats1)
                feats2 = block(feats2)
            else:
                raise NotImplementedError

        return feats1, feats2
