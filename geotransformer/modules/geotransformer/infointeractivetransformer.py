# import torch
# import torch.nn as nn

# from geotransformer.modules.transformer.information_interactive import InformationInteractive
# from geotransformer.modules.transformer import SinusoidalPositionalEmbedding
# from geotransformer.modules.ops import pairwise_distance
# import numpy as np


# class GeometricStructureEmbedding(nn.Module):
#     def __init__(self, hidden_dim, sigma_d, sigma_a, angle_k, reduction_a='max'):
#         super().__init__()
#         self.sigma_d = sigma_d
#         self.sigma_a = sigma_a
#         self.factor_a = 180.0 / (self.sigma_a * np.pi)
#         self.angle_k = angle_k

#         self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
#         self.proj_d = nn.Linear(hidden_dim, hidden_dim)
#         self.proj_a = nn.Linear(hidden_dim, hidden_dim)

#         if reduction_a not in ['max', 'mean']:
#             raise ValueError(f'Unsupported reduction mode: {reduction_a}')
#         self.reduction_a = reduction_a

#     @torch.no_grad()
#     def get_embedding_indices(self, points):
#         B, N, _ = points.shape
#         dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)
#         d_indices = dist_map / self.sigma_d

#         k = self.angle_k
#         knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
#         knn_indices = knn_indices.unsqueeze(3).expand(B, N, k, 3)
#         expanded_points = points.unsqueeze(1).expand(B, N, N, 3)
#         knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)

#         ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)
#         anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
#         ref_vectors = ref_vectors.unsqueeze(2).expand(B, N, N, k, 3)
#         anc_vectors = anc_vectors.unsqueeze(3).expand(B, N, N, k, 3)

#         sin = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)
#         cos = torch.sum(ref_vectors * anc_vectors, dim=-1)
#         angles = torch.atan2(sin, cos)  # (B, N, N, k)
#         a_indices = angles * self.factor_a

#         return d_indices, a_indices

#     def forward(self, points):
#         d_indices, a_indices = self.get_embedding_indices(points)

#         d_embed = self.proj_d(self.embedding(d_indices))
#         a_embed = self.proj_a(self.embedding(a_indices))

#         if self.reduction_a == 'max':
#             a_embed = a_embed.max(dim=3)[0]
#         else:
#             a_embed = a_embed.mean(dim=3)

#         return d_embed + a_embed


# class InfoInteractiveTransformer(nn.Module):
#     def __init__(
#         self,
#         input_dim,
#         output_dim,
#         feat_dims,
#         gcn_k,
#         ppf_k,
#         radius,
#         bottleneck,
#         nhead,
#         layer_names,
#         sigma_d,
#         sigma_a,
#         angle_k,
#         reduction_a='max'
#     ):
#         super().__init__()
        
#         self.embedding = GeometricStructureEmbedding(
#             feat_dims, 
#             sigma_d, 
#             sigma_a, 
#             angle_k, 
#             reduction_a
#         )
        
#         self.in_proj = nn.Linear(input_dim, feat_dims)
#         self.out_proj = nn.Linear(feat_dims, output_dim)

#         self.info = InformationInteractive(
#             layer_names=layer_names,
#             feat_dims=feat_dims,
#             gcn_k=gcn_k,
#             ppf_k=ppf_k,
#             radius=radius,
#             bottleneck=bottleneck,
#             nhead=nhead
#         )

#     def forward(
#         self,
#         ref_points,  # (B, N, 3)
#         src_points,  # (B, M, 3)
#         ref_feats,   # (B, N, C)
#         src_feats,   # (B, M, C)
#         ref_normals, # (B, N, 3)
#         src_normals  # (B, M, 3)
#     ):
#         # Geometric positional embedding (currently unused, but can be added to feats if needed)
#         ref_pos_embed = self.embedding(ref_points.transpose(1, 2))  # input: (B, N, 3)
#         src_pos_embed = self.embedding(src_points.transpose(1, 2))  # input: (B, M, 3)

#         # Feature projection
#         ref_feats = self.in_proj(ref_feats + ref_pos_embed).permute(0, 2, 1).contiguous()
#         src_feats = self.in_proj(src_feats + src_pos_embed).permute(0, 2, 1).contiguous()
        
#         # Convert to (B, 3, N)
#         ref_points = ref_points.permute(0, 2, 1).contiguous()
#         src_points = src_points.permute(0, 2, 1).contiguous()
#         ref_normals = ref_normals.permute(0, 2, 1).contiguous()
#         src_normals = src_normals.permute(0, 2, 1).contiguous()

#         # Information interaction
#         ref_feats, src_feats = self.info(
#             ref_points, ref_feats,
#             src_points, src_feats,
#             ref_normals, src_normals
#         )

#         # Output projection
#         ref_feats = self.out_proj(ref_feats.permute(0, 2, 1).contiguous())  # (B, N, C)
#         src_feats = self.out_proj(src_feats.permute(0, 2, 1).contiguous())  # (B, M, C)

#         return ref_feats, src_feats
import torch
import torch.nn as nn
import numpy as np

from geotransformer.modules.transformer.information_interactive import InformationInteractive
from geotransformer.modules.transformer import SinusoidalPositionalEmbedding
from geotransformer.modules.ops import pairwise_distance


class GeometricStructureEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d, sigma_a, angle_k, reduction_a='max'):
        super().__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = angle_k

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)

        if reduction_a not in ['max', 'mean']:
            raise ValueError(f'Unsupported reduction mode: {reduction_a}')
        self.reduction_a = reduction_a

    @torch.no_grad()
    def get_embedding_indices(self, points):
        B, N, _ = points.shape
        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)
        d_indices = dist_map / self.sigma_d

        k = self.angle_k
        if k >= N:
            raise ValueError(f"angle_k ({k}) must be smaller than number of points ({N}).")

        knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
        knn_indices = knn_indices.unsqueeze(3).expand(B, N, k, 3)
        expanded_points = points.unsqueeze(1).expand(B, N, N, 3)
        knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)

        ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)
        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
        ref_vectors = ref_vectors.unsqueeze(2).expand(B, N, N, k, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(B, N, N, k, 3)

        sin = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)
        cos = torch.sum(ref_vectors * anc_vectors, dim=-1)
        angles = torch.atan2(sin, cos)  # (B, N, N, k)
        a_indices = angles * self.factor_a

        return d_indices, a_indices

    def forward(self, points):  # points: (B, N, 3)
        d_indices, a_indices = self.get_embedding_indices(points)

        d_embed = self.proj_d(self.embedding(d_indices))       # (B, N, N, C)
        a_embed = self.proj_a(self.embedding(a_indices))       # (B, N, N, k, C)

        if self.reduction_a == 'max':
            a_embed = a_embed.max(dim=3)[0]  # (B, N, N, C)
        else:
            a_embed = a_embed.mean(dim=3)

        return d_embed + a_embed  # (B, N, N, C)


class InfoInteractiveTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        feat_dims,
        gcn_k,
        ppf_k,
        radius,
        bottleneck,
        nhead,
        layer_names,
        sigma_d,
        sigma_a,
        angle_k,
        reduction_a='max'
    ):
        super().__init__()

        self.embedding = GeometricStructureEmbedding(
            hidden_dim=feat_dims,
            sigma_d=sigma_d,
            sigma_a=sigma_a,
            angle_k=angle_k,
            reduction_a=reduction_a
        )

        self.in_proj = nn.Linear(input_dim, feat_dims)
        self.out_proj = nn.Linear(feat_dims, output_dim)

        self.info = InformationInteractive(
            layer_names=layer_names,
            feat_dims=feat_dims,
            gcn_k=gcn_k,
            ppf_k=ppf_k,
            radius=radius,
            bottleneck=bottleneck,
            nhead=nhead
        )

    def forward(
        self,
        ref_points,  # (B, N, 3)
        src_points,  # (B, M, 3)
        ref_feats,   # (B, N, C)
        src_feats,   # (B, M, C)
        ref_normals, # (B, N, 3)
        src_normals  # (B, M, 3)
    ):

        # Positional embedding
        ref_pos_embed = self.embedding(ref_points)  # (B, N, N, C)
        src_pos_embed = self.embedding(src_points)  # (B, M, M, C)

        # Skip connection trick: Use diagonal for positional enhancement
        ref_pos_embed = torch.diagonal(ref_pos_embed, dim1=1, dim2=2).transpose(1, 2)  # (B, N, C)
        src_pos_embed = torch.diagonal(src_pos_embed, dim1=1, dim2=2).transpose(1, 2)  # (B, M, C)

        # Feature projection
        ref_feats = self.in_proj(ref_feats) + ref_pos_embed  # (B, N, C)
        src_feats = self.in_proj(src_feats) + src_pos_embed  # (B, M, C)
        
        # ref_feats = self.in_proj(ref_feats) # (B, N, C)
        # src_feats = self.in_proj(src_feats) # (B, M, C)

        ref_feats = ref_feats.permute(0, 2, 1).contiguous()  # (B, C, N)
        src_feats = src_feats.permute(0, 2, 1).contiguous()  # (B, C, M)

        # Transpose coordinates and normals
        ref_points = ref_points.permute(0, 2, 1).contiguous()
        src_points = src_points.permute(0, 2, 1).contiguous()
        ref_normals = ref_normals.permute(0, 2, 1).contiguous()
        src_normals = src_normals.permute(0, 2, 1).contiguous()

        # Main interaction
        ref_feats, src_feats = self.info(
            ref_points, ref_feats,
            src_points, src_feats,
            ref_normals, src_normals
        )

        # Output projection
        ref_feats = self.out_proj(ref_feats.permute(0, 2, 1).contiguous())  # (B, N, C)
        src_feats = self.out_proj(src_feats.permute(0, 2, 1).contiguous())  # (B, M, C)

        return ref_feats, src_feats
