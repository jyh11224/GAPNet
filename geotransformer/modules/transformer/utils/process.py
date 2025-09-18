import math
import numpy as np
import torch


def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def fmat(arr):
    return np.around(arr,3)


def to_tensor(x, use_cuda):
    if use_cuda:
        return torch.tensor(x).cuda()
    else:
        return torch.tensor(x)


# def gather_points(points, inds):
#     '''

#     :param points: shape=(B, N, C)
#     :param inds: shape=(B, M) or shape=(B, M, K)
#     :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
#     '''
#     device = points.device
#     B, N, C = points.shape
#     inds_shape = list(inds.shape)
#     inds_shape[1:] = [1] * len(inds_shape[1:])
#     repeat_shape = list(inds.shape)
#     repeat_shape[0] = 1
#     batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
#     return points[batchlists, inds, :]
def gather_points(points, inds):
    inds = inds.to(points.device)
    B = points.shape[0]
    batchlists = torch.arange(B, dtype=torch.long, device=points.device).view(B, 1, 1)
    batchlists = batchlists.expand(-1, inds.shape[1], inds.shape[2])
    
    return points[batchlists, inds, :]


def square_dists(points1, points2):
    '''
    Calculate square dists between two group points
    :param points1: shape=(B, N, C)
    :param points2: shape=(B, M, C)
    :return:
    '''
    B, N, C = points1.shape
    _, M, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, N, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, M)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.clamp(dists, min=1e-8)
    return dists.float()


def ball_query(xyz, new_xyz, radius, K, rt_density=False):
    '''
    :param xyz: shape=(B, N, 3)
    :param new_xyz: shape=(B, M, 3)
    :param radius: int
    :param K: int, an upper limit samples
    :return: shape=(B, M, K)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    M = new_xyz.shape[1]
    grouped_inds = torch.arange(0, N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, M, 1)
    dists = square_dists(new_xyz, xyz)
    grouped_inds[dists > radius ** 2] = N
    if rt_density:
        density = torch.sum(grouped_inds < N, dim=-1)
        density = density / N
    grouped_inds = torch.sort(grouped_inds, dim=-1)[0][:, :, :K]
    grouped_min_inds = grouped_inds[:, :, 0:1].repeat(1, 1, min(K, grouped_inds.size(2)))
    grouped_inds[grouped_inds == N] = grouped_min_inds[grouped_inds == N]
    if rt_density:
        return grouped_inds, density
    return grouped_inds


# def sample_and_group(xyz, points, M, radius, K, use_xyz=True, rt_density=False):
#     '''
#     :param xyz: shape=(B, N, 3)
#     :param points: shape=(B, N, C)
#     :param M: int
#     :param radius:float
#     :param K: int
#     :param use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
#     :return: new_xyz, shape=(B, M, 3); new_points, shape=(B, M, K, C+3);
#              group_inds, shape=(B, M, K); grouped_xyz, shape=(B, M, K, 3)
#     '''
#     assert M < 0
#     new_xyz = xyz
#     if rt_density:
#         grouped_inds, density = ball_query(xyz, new_xyz, radius, K,
#                                            rt_density=True)
#     else:
#         grouped_inds = ball_query(xyz, new_xyz, radius, K, rt_density=False)
#     grouped_xyz = gather_points(xyz, grouped_inds)
#     grouped_xyz -= torch.unsqueeze(new_xyz, 2).repeat(1, 1, min(K, grouped_inds.size(2)), 1)
#     if points is not None:
#         grouped_points = gather_points(points, grouped_inds)
#         if use_xyz:
#             new_points = torch.cat((grouped_xyz.float().to(grouped_points.device), grouped_points.float()), dim=-1)
#         else:
#             new_points = grouped_points
#     else:
#         new_points = grouped_xyz
#     if rt_density:
#         return new_xyz, new_points, grouped_inds, grouped_xyz, density
#     return new_xyz, new_points, grouped_inds, grouped_xyz
def sample_and_group(xyz, points, M, radius, K, use_xyz=True, rt_density=False):
    '''
    xyz: (B, N, 3)
    points: (B, N, C)
    M: number of centroids to sample
    radius: neighborhood radius
    K: number of neighbors to group
    '''
    B, N, _ = xyz.shape
    device = xyz.device

    # 1. 샘플링 (FPS 등). FPS 함수가 없으면 임시로 처음 M개 선택
    # TODO: FPS 함수 적용 권장
    if M <= 0 or M > N:
        new_xyz = xyz  # 전체 포인트 사용
    else:
        new_xyz = xyz[:, :M, :]

    if rt_density:
        grouped_inds, density = ball_query(xyz, new_xyz, radius, K, rt_density=True)
    else:
        grouped_inds = ball_query(xyz, new_xyz, radius, K, rt_density=False)

    grouped_inds = ball_query(xyz, new_xyz, radius, K, rt_density=False)

    # 디바이스 맞추기
    device = xyz.device
    if grouped_inds.device != device:
        grouped_inds = grouped_inds.to(device)

    grouped_xyz = gather_points(xyz, grouped_inds)  # (B, M, K, 3)
    grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)

    if points is not None:
        if points.device != device:
            points = points.to(device)
        grouped_points = gather_points(points, grouped_inds)

        # 둘 다 device 통일
        grouped_xyz = grouped_xyz.to(device)
        grouped_points = grouped_points.to(device)

        if use_xyz:
            new_points = torch.cat((grouped_xyz.float(), grouped_points.float()), dim=-1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    if rt_density:
        return new_xyz, new_points, grouped_inds, grouped_xyz, density
    else:
        return new_xyz, new_points, grouped_inds, grouped_xyz



# def angle(v1: torch.Tensor, v2: torch.Tensor):
#     """Compute angle between 2 vectors
#     For robustness, we use the same formulation as in PPFNet, i.e.
#         angle(v1, v2) = atan2(cross(v1, v2), dot(v1, v2)).
#     This handles the case where one of the vectors is 0.0, since torch.atan2(0.0, 0.0)=0.0
#     Args:
#         v1: (B, *, 3)
#         v2: (B, *, 3)
#     Returns:
#     """

#     cross_prod = torch.stack([v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1],
#                               v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2],
#                               v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]], dim=-1)
#     cross_prod_norm = torch.norm(cross_prod, dim=-1)
#     dot_prod = torch.sum(v1 * v2, dim=-1)

#     return torch.atan2(cross_prod_norm, dot_prod)
def angle(v1: torch.Tensor, v2: torch.Tensor):
    """Compute angle between 2 vectors using atan2(norm(cross), dot)

    Args:
        v1: (..., 3)
        v2: (..., 3)
    Returns:
        angle: (...) in radians
    """
    
    cross_prod = torch.stack([
        v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1],
        v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2],
        v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    ], dim=-1)

    cross_norm = torch.norm(cross_prod, dim=-1)  # (...,)
    dot_prod = torch.sum(v1 * v2, dim=-1)        # (...,)
    return torch.atan2(cross_norm, dot_prod)
