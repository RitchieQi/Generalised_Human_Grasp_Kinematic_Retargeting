import numpy as np
import cv2
import torch
from torch.nn import functional as F
import time
import os
import sys
import json
from torch import distributed as dist

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    """
    @description: Modified from https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/blob/master/data/dataset.py.
                  get affine transform matrix
    ---------
    @param: image center, original image size, desired image size, scale factor, rotation degree, whether to get inverse transformation.
    -------
    @Returns: affine transformation matrix
    -------
    """

    def rotate_2d(pt_2d, rot_rad):
        x = pt_2d[0]
        y = pt_2d[1]
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        xx = x * cs - y * sn
        yy = x * sn + y * cs
        return np.array([xx, yy], dtype=np.float32)

    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def generate_patch_image(cvimg, bbox, input_shape, scale, rot):
    """
    @description: Modified from https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/blob/master/data/dataset.py.
                  generate the patch image from the bounding box and other parameters.
    ---------
    @param: input image, bbox(x1, y1, h, w), dest image shape, do_flip, scale factor, rotation degrees.
    -------
    @Returns: processed image, affine_transform matrix to get the processed image.
    -------
    """

    img = cvimg.copy()
    img_height, img_width, _ = img.shape

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, input_shape[1], input_shape[0], scale, rot, inv=False)
    img_patch = cv2.warpAffine(img, trans, (int(input_shape[1]), int(input_shape[0])), flags=cv2.INTER_LINEAR)
    new_trans = np.zeros((3, 3), dtype=np.float32)
    new_trans[:2, :] = trans
    new_trans[2, 2] = 1

    return img_patch, new_trans

def merge_handobj_bbox(hand_bbox, obj_bbox):
    # the format of bbox: xyxy
    tl = np.min(np.concatenate([hand_bbox.reshape((2, 2)), obj_bbox.reshape((2, 2))], axis=0), axis=0)
    br = np.max(np.concatenate([hand_bbox.reshape((2, 2)), obj_bbox.reshape((2, 2))], axis=0), axis=0)
    box_size = br - tl
    bbox = np.concatenate([tl, box_size], axis=0)

    return bbox
    

def process_bbox(bbox):
    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    aspect_ratio = 1.
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * 1.25
    bbox[3] = h * 1.25
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.

    return bbox

class PerspectiveCamera:
    def __init__(self, fx, fy, cx, cy, R=np.eye(3), t=np.zeros(3)):
        self.K = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]], dtype=np.float32)

        self.R = np.array(R, dtype=np.float32).copy()
        assert self.R.shape == (3, 3)

        self.t = np.array(t, dtype=np.float32).copy()
        assert self.t.size == 3
        self.t = self.t.reshape(3, 1)

    def update_virtual_camera_after_crop(self, bbox, option='same'):
        left, upper, width, height = bbox
        new_img_center = np.array([left + width / 2, upper + height / 2, 1], dtype=np.float32).reshape(3, 1)
        new_cam_center = np.linalg.inv(self.K[:3, :3]).dot(new_img_center)
        self.K[0, 2], self.K[1, 2] = width / 2, height / 2

        x, y, z = new_cam_center[0], new_cam_center[1], new_cam_center[2]
        sin_theta = -y / np.sqrt(1 + x ** 2 + y ** 2)
        cos_theta = np.sqrt(1 + x ** 2) / np.sqrt(1 + x ** 2 + y ** 2)
        R_x = np.array([[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]], dtype=np.float32)
        sin_phi = x / np.sqrt(1 + x ** 2)
        cos_phi = 1 / np.sqrt(1 + x ** 2)
        R_y = np.array([[cos_phi, 0, sin_phi], [0, 1, 0], [-sin_phi, 0, cos_phi]], dtype=np.float32)
        self.R = R_y @ R_x

        # update focal length for virtual camera; please refer to the paper "PCLs: Geometry-aware Neural Reconstruction of 3D Pose with Perspective Crop Layers" for more details.
        if option == 'length':
            self.K[0, 0] = self.K[0, 0] * np.sqrt(1 + x ** 2 + y ** 2)
            self.K[1, 1] = self.K[1, 1] * np.sqrt(1 + x ** 2 + y ** 2)
        
        if option == 'scale':
            self.K[0, 0] = self.K[0, 0] * np.sqrt(1 + x ** 2 + y ** 2) * np.sqrt(1 + x ** 2)
            self.K[1, 1] = self.K[1, 1] * (1 + x ** 2 + y ** 2)/ np.sqrt(1 + x ** 2)

    def update_intrinsics_after_crop(self, bbox):
        left, upper, _, _ = bbox

        cx, cy = self.K[0, 2], self.K[1, 2]

        new_cx = cx - left
        new_cy = cy - upper

        self.K[0, 2], self.K[1, 2] = new_cx, new_cy

    def update_intrinsics_after_resize(self, image_shape, new_image_shape):
        height, width = image_shape
        new_height, new_width = new_image_shape

        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]

        new_fx = fx * (new_width / width)
        new_fy = fy * (new_height / height)
        new_cx = cx * (new_width / width)
        new_cy = cy * (new_height / height)

        self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2] = new_fx, new_fy, new_cx, new_cy

    @property
    def projection(self):
        return self.K.dot(self.extrinsics)

    @property
    def intrinsics(self):
        return self.K

    @property
    def extrinsics(self):
        return np.hstack([self.R, self.t])


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