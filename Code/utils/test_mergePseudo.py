import kornia
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import random

from scipy.ndimage import distance_transform_edt


def get_edge(label, kernel_size=3, iterations=3, decay_rate=0.1):
    """
    对01标签进行边缘向内的数值递减羽化
    :param label: 输入的01标签张量，形状为 (C, H, W)
    :param kernel_size: 形态学操作的卷积核大小
    :param iterations: 形态学操作的迭代次数
    :param decay_rate: 像素值递减的速率
    :return: 羽化后的标签张量
    """
    # 进行膨胀操作，找出标签的边缘
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=label.device)
    dilated = F.conv2d(label.float(), kernel, padding=kernel_size // 2)
    dilated = (dilated > 0).float()

    # 进行腐蚀操作，得到内部区域
    eroded = F.conv2d(label.float(), kernel, padding=kernel_size // 2, stride=1)
    eroded = (eroded == kernel_size * kernel_size).float()

    # 计算边缘
    edge = dilated - eroded



    return edge

def generate_irregular_circle_label(size=(256, 256), center=None, radius=None, irregularity=0.1, spikiness=0.1, num_points=30):
    """
    生成带有不规则变化的近似圆形0 - 1伪标签
    :param size: 标签的尺寸 (高度, 宽度)
    :param center: 圆心坐标 (x, y)，默认为图像中心
    :param radius: 圆的半径，默认为图像短边的一半
    :param irregularity: 不规则程度，取值范围 [0, 1]
    :param spikiness: 尖刺程度，取值范围 [0, 1]
    :param num_points: 用于定义圆形的点数
    :return: 0 - 1伪标签
    """
    height, width = size
    if center is None:
        center = (width // 2, height // 2)
    if radius is None:
        radius = min(width, height) // 4

    # 生成不规则的点
    angles = np.linspace(0, 2 * np.pi, num_points)
    points = []
    for angle in angles:
        r = radius * (1 + random.uniform(-irregularity, irregularity))
        r = r * (1 + spikiness * np.sin(random.uniform(0, 2 * np.pi)))
        x = int(center[0] + r * np.cos(angle))
        y = int(center[1] + r * np.sin(angle))
        points.append([x, y])
    points = np.array(points, dtype=np.int32)

    # 创建空白图像
    label = np.zeros((height, width), dtype=np.uint8)

    # 绘制不规则圆形
    cv2.fillPoly(label, [points], 1)

    return torch.from_numpy(label).unsqueeze(0).unsqueeze(0).unsqueeze(-1)

def fuse_masks(mask_a: torch.Tensor, mask_b: torch.Tensor, bias: float) -> torch.Tensor:
    """
    Args:
        mask_a: Tensor of shape [B, 1, H, W], dtype=torch.bool
        mask_b: Tensor of same shape
        bias: float between 0 and 1. 0.5 means center between the two.
    Returns:
        fused_mask: Tensor of shape [B, 1, H, W], dtype=torch.bool
    """
    assert mask_a.shape == mask_b.shape
    # assert mask_a.dtype == torch.bool and mask_b.dtype == torch.bool
    assert 0 <= bias <= 1

    # Convert bool masks to float for distance transform
    mask_a_f = mask_a.float()
    mask_b_f = mask_b.float()

    # Compute distance to the boundary (inside: positive, outside: negative)
    # 计算数组中值为 非零 的点到最近 零值点 的距离，并同时返回最近非零值点的索引
    dist_a, nearest_indices = distance_transform_edt(mask_a_f, return_indices=True)

    # 计算数组中值为 0 的点到最近 非零值点 的距离，并同时返回最近非零值点的索引
    dist_b, nearest_indices = distance_transform_edt(mask_b_f, return_indices=True)

    dist_a = kornia.contrib.distance_transform(mask_a_f)  # inside: >0
    dist_b = kornia.contrib.distance_transform(mask_b_f)

    # Normalize distance (optional, depending on desired behavior)
    # dist_a = dist_a / (dist_a.max(dim=(-1, -2), keepdim=True).values + 1e-6)
    # dist_b = dist_b / (dist_b.max(dim=(-1, -2), keepdim=True).values + 1e-6)

    # Bias fusion: greater distance wins
    fused_score = (1 - bias) * dist_a - bias * dist_b
    fused_mask = fused_score > 0

    return fused_mask

if __name__ == '__main__':
    # 示例使用
    label1,label2 = generate_irregular_circle_label(),generate_irregular_circle_label()
    edge1,edge2 = get_edge(label1),get_edge(label2)
    out = fuse_masks(label1,label2,bias=0.5)
    print('OK')


