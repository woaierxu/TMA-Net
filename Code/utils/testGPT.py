import time
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt

# from code_MQX.utils.test_mergePseudo import generate_irregular_circle_label, get_edge


def compute_signed_distance_3d(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute signed distance transform for 3D masks.

    Args:
        mask: [B, 1, D, H, W] binary tensor (bool or 0/1 float)

    Returns:
        sdf: [B, 1, D, H, W] signed float tensor
    """
    B = mask.shape[0]
    sdf_list = []

    for b in range(B):
        m = mask[b, 0].detach().cpu().numpy().astype(np.uint8)
        pos_dist = distance_transform_edt(m == 0)
        neg_dist = distance_transform_edt(m == 1)
        sdf = pos_dist - neg_dist
        sdf_list.append(torch.from_numpy(sdf).unsqueeze(0))  # [1, D, H, W]

    return torch.stack(sdf_list, dim=0).to(mask.device)  # [B, 1, D, H, W]


def fuse_label_by_sdf_3d(mask_a: torch.Tensor, mask_b: torch.Tensor, bias: float) -> torch.Tensor:
    """
    SDF fusion for 3D masks: input [B, 1, D, H, W], returns same shape bool mask.
    """
    assert mask_a.shape == mask_b.shape
    assert 0 <= bias <= 1

    sdf_a = compute_signed_distance_3d(mask_a)
    sdf_b = compute_signed_distance_3d(mask_b)

    fused_sdf = (1 - bias) * sdf_a + bias * sdf_b
    fused_mask = fused_sdf < 0
    return fused_mask.float()

# if __name__ == '__main__':
#     label1,label2 = generate_irregular_circle_label((112,112)),generate_irregular_circle_label((112,112))
#     label1,label2 = label1.repeat(1,1,1,1,80),label2.repeat(1,1,1,1,80)
#
#     # edge1,edge2 = get_edge(label1),get_edge(label2)
#     t1 = time.time()
#     fused = fuse_label_by_sdf_3d(label1, label2, bias=0.4)
#     t2 = time.time()
#     # edge_fuse = get_edge(fused)
#     print(f'OK:{t2-t1}')