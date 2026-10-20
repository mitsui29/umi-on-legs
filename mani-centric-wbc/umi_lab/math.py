"""Torch rotation utilities. Quaternions use Isaac Lab's wxyz convention."""

import torch


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), -1)


def axis_angle_to_quat(v):
    angle = v.norm(dim=-1, keepdim=True)
    return torch.cat((torch.cos(angle / 2), 0.5 * torch.sinc(angle / (2 * torch.pi)) * v), -1)


def euler_xyz_to_quat(v):
    """Match PyTorch3D euler_angles_to_matrix(..., 'XYZ'): Rx @ Ry @ Rz."""
    c, s = torch.cos(v / 2), torch.sin(v / 2)
    x, y, z = c.unbind(-1)
    a, b, d = s.unbind(-1)
    return torch.stack((x*y*z-a*b*d, a*y*z+x*b*d, x*b*z-a*y*d, x*y*d+a*b*z), -1)


def quat_to_matrix(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def rotation_angle(a, b):
    relative = a @ b.transpose(-1, -2)
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return cosine.clamp(-1, 1).acos()


def relative_pose_observation(pos, rot, reference_pos, reference_rot):
    """Original layout: all positions, then all first-two-row rotation 6D values."""
    inv = reference_rot.transpose(-1, -2)[:, None]
    local_pos = (inv @ (pos - reference_pos[:, None])[..., None]).squeeze(-1)
    local_rot = inv @ rot
    return torch.cat((10 * local_pos.flatten(1), 1.5 * local_rot[..., :2, :].flatten(1)), -1)
