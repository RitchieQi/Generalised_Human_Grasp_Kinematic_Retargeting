import numpy as np
import cv2
import torch
from torch.nn import functional as F
import time
import os
import sys
import json
from torch import distributed as dist

def soft_argmax(heatmap_size, heatmaps, num_joints):
    depth_dim = heatmaps.shape[1] // num_joints
    H_heatmaps = heatmaps.shape[2]
    W_heatmaps = heatmaps.shape[3]
    heatmaps = heatmaps.reshape((-1, num_joints, depth_dim * H_heatmaps * W_heatmaps))
    heatmaps = F.softmax(heatmaps, 2)
    confidence, _ = torch.max(heatmaps, 2)
    heatmaps = heatmaps.reshape((-1, num_joints, depth_dim, H_heatmaps, W_heatmaps))

    accu_x = heatmaps.sum(dim=(2, 3))
    accu_y = heatmaps.sum(dim=(2, 4))
    accu_z = heatmaps.sum(dim=(3, 4))

    accu_x = accu_x * torch.arange(heatmap_size[1]).float().to(heatmaps.device)[None, None, :]
    accu_y = accu_y * torch.arange(heatmap_size[0]).float().to(heatmaps.device)[None, None, :]
    accu_z = accu_z * torch.arange(heatmap_size[2]).float().to(heatmaps.device)[None, None, :]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)

    return coord_out, confidence


def decode_volume(image_size, heatmap_size, depth_dim, heatmaps, center3d, cam_intr):
    hm_pred = heatmaps.clone()
    hm_pred[:, :, 0] *= (image_size[1] // heatmap_size[1])
    hm_pred[:, :, 1] *= (image_size[0] // heatmap_size[0])
    hm_pred[:, :, 2] = (hm_pred[:, :, 2] / heatmap_size[2] * 2 - 1) * depth_dim + center3d[:, [2]]

    fx = cam_intr[:, 0, 0].unsqueeze(1)
    fy = cam_intr[:, 1, 1].unsqueeze(1)
    cx = cam_intr[:, 0, 2].unsqueeze(1)
    cy = cam_intr[:, 1, 2].unsqueeze(1)

    cam_x = ((hm_pred[:, :, 0] - cx) / fx * hm_pred[:, :, 2]).unsqueeze(2)
    cam_y = ((hm_pred[:, :, 1] - cy) / fy * hm_pred[:, :, 2]).unsqueeze(2)
    cam_z = hm_pred[:, :, [2]]
    cam_coords = torch.cat([cam_x, cam_y, cam_z], 2)

    return cam_coords

def recover_3d_proj(objpoints3d, camintr, est_scale, est_trans, off_z=0.4, input_res=(256, 256)):
    focal = camintr[:, :1, :1]
    batch_size = objpoints3d.shape[0]
    focal = focal.view(batch_size, 1)
    est_scale = est_scale.view(batch_size, 1)
    est_trans = est_trans.view(batch_size, 2)
    # est_scale is homogeneous to object scale change in pixels
    est_Z0 = focal * est_scale + off_z
    cam_centers = camintr[:, :2, 2]
    img_centers = (cam_centers.new(input_res) / 2).view(1, 2).repeat(batch_size, 1)
    est_XY0 = (est_trans + img_centers - cam_centers) * est_Z0 / focal
    est_c3d = torch.cat([est_XY0, est_Z0], -1).unsqueeze(1)
    recons3d = est_c3d + objpoints3d
    return recons3d, est_c3d


def get_mano_preds(mano_results, image_size, cam_intr, wrist_pos=None):
    if mano_results['scale_trans'] is not None:
        trans = mano_results['scale_trans'][:, 1:]
        scale = mano_results['scale_trans'][:, [0]]
        final_trans = trans * 100.0
        final_scale = scale * 0.0001
        cam_joints, center3d = recover_3d_proj(mano_results['joints'], cam_intr, final_scale, final_trans, input_res=image_size)
        cam_verts = center3d + mano_results['verts']
        mano_results['joints'] = cam_joints
        mano_results['verts'] = cam_verts
    else:
        center3d = wrist_pos.reshape((mano_results['joints'].shape[0], 1, 3))
        mano_results['joints'] = mano_results['joints'] + center3d
        mano_results['verts'] = mano_results['verts'] + center3d
    
    return mano_results




class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.warm_up = 0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        if self.warm_up < 10:
            self.warm_up += 1
            return self.diff
        else:
            self.total_time += self.diff
            self.calls += 1
            self.average_time = self.total_time / self.calls

        if average:
            return self.average_time
        else:
            return self.diff

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def export_pose_results(path, pose_result, metas):
    if isinstance(pose_result, list):
        num_frames = len(pose_result)
        for i in range(num_frames):
            sample_id = metas['id'][i][0]
            with open(os.path.join(path, sample_id + '.json'), 'w') as f:
                output = dict()
                output['cam_extr'] = metas['cam_extr'][i][0][:3, :3].cpu().numpy().tolist()
                output['cam_intr'] = metas['cam_intr'][i][0][:3, :3].cpu().numpy().tolist()
                if pose_result[i] is not None:
                    for key in pose_result[i].keys():
                        if pose_result[i][key] is not None:
                            output[key] = pose_result[i][key][0].cpu().numpy().tolist()
                        else:
                            continue
                json.dump(output, f)
    else:
        sample_id = metas['id'][0]
        with open(os.path.join(path, sample_id + '.json'), 'w') as f:
            output = dict()
            output['cam_extr'] = metas['cam_extr'][0][:3, :3].cpu().numpy().tolist()
            output['cam_intr'] = metas['cam_intr'][0][:3, :3].cpu().numpy().tolist()
            if pose_result is not None:
                for key in pose_result.keys():
                    if pose_result[key] is not None:
                        output[key] = pose_result[key][0].cpu().numpy().tolist()
                    else:
                        continue
            json.dump(output, f)
            
def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt