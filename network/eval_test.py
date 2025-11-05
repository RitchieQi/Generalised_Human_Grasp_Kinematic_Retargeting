import os
import os.path as osp
import sys
import torch
import numpy as np
import json


def main():
    hand_pose_dir = osp.join(osp.dirname(__file__), 'hmano_osdf', 'hand_pose_results')
    obj_pose_dir = osp.join(osp.dirname(__file__), 'hmano_osdf', 'obj_pose_results')
    
    anno = osp.join(osp.dirname(__file__), '..', 'CtcSDF', 'data', 'dexycb_test_s0.json')
    
    # Load the annotations
    with open(anno, 'r') as f:
        annotations = json.load(f)
    
    