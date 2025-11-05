import argparse
import yaml
import json
import numpy as np
import os

import trimesh

import cv2
import pyrender
from dataset import CtcSDF_dataset

def deproject_points(points, cam_intr, cam_extr):
    points_3d = np.asarray(points)
    points_3d = np.concatenate((points_3d, np.ones((points_3d.shape[0], 1))), axis=1)
    cam_extr_inv = np.linalg.inv(cam_extr)
    points_3d_cam = np.dot(cam_extr_inv, points_3d.T).T
    points_3d_cam = points_3d_cam[:, :3] / points_3d_cam[:, 3:]
    points_3d_cam = np.concatenate((points_3d_cam, np.ones((points_3d_cam.shape[0], 1))), axis=1)
    points_2d = np.dot(cam_intr, points_3d_cam.T).T
    points_2d = points_2d[:, :2] / points_2d[:, 2:]
    return points_2d
    

def main():
    from torch.utils.data import DataLoader
    dataset = CtcSDF_dataset(task='train')
    DataLoader = DataLoader(dataset, batch_size=1, shuffle=False)
    inputs, targets, metas = next(iter(DataLoader))
    color = inputs['img'].squeeze(0).numpy()
    # color = np.clip(color, 0, 1)
    # print('color', color)
    # print('color', color.shape)

    hand_joints = targets['hand_joints_3d'].squeeze().numpy()
    obj_center = targets['obj_center_3d'].squeeze().numpy()
    cam_intr = metas['cam_intr'].squeeze().numpy()
    cam_intr = np.concatenate((cam_intr, np.array([[0, 0, 0, 1]])), axis=0)
    cam_extr = metas['cam_extr'].squeeze().numpy()
    cam_extr = np.concatenate((cam_extr, np.array([[0, 0, 0, 1]])), axis=0)
    # print('cam_intr', cam_intr)
    # print('cam_extr', cam_extr)
    # color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    point_2d = deproject_points(hand_joints, cam_intr, cam_extr)
    for (u, v) in point_2d:
        u, v = int(round(u)), int(round(v))
        if 0 <= u < color.shape[1] and 0 <= v < color.shape[0]:
            cv2.circle(color, (u, v), radius=5, color=(0, 0, 255), thickness=-1)
    
    # cv2.imshow('image', color)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
    
    
    
        