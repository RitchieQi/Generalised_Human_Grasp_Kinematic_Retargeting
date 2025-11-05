import os
osp = os.path
import numpy as np
from typing import List, Optional, Tuple, Union, Iterator, Any, Dict
from dex_ycb_toolkit.dex_ycb import DexYCBDataset
import json
from scipy.spatial.transform import Rotation as R
import torch
import time
import trimesh as tm
from torch.utils.data import Dataset
import cv2
from manopth.manolayer import ManoLayer
import traceback
import os
from optimisation_tmp.utils import rodrigues


class ycb_opt_fetcher(Dataset):
    def __init__(self, 
                 dexycb_dir = '/home/liyuan/DexYCB/', 
                 task = "test", 
                 data_dir = osp.join(osp.dirname(osp.abspath(__file__)),"..","..", "CtcSDF", "data"), 
                 data_sample = 5,
                 robot_name = "Shadow",
                 mu = 0.1,
                 SDF_source = False,
                 repeat = 1,
                 load_assert = False,
                 exp_name = "genhand",
                 test_sdf = False,
                ):
        self.sdf_source = SDF_source
        os.environ["DEX_YCB_DIR"] = dexycb_dir
        self.getdata = DexYCBDataset("s0", task)
        self.robot_name = robot_name
        self.mu = mu
        self.data_sample = data_sample
        if exp_name == "genhand" and test_sdf == False:
            self.result_dir = osp.join(osp.abspath(osp.dirname(__file__)), "..", "..", "results", self.robot_name, "mu"+str(mu))
        elif exp_name == "nv" and test_sdf == False:
            self.result_dir = osp.join(osp.abspath(osp.dirname(__file__)), "..", "..", "results_nv", self.robot_name, "mu"+str(mu))
        elif task == "train":
            self.result_dir = osp.join(osp.abspath(osp.dirname(__file__)), "..", "..")
        elif task == "test" and exp_name is None:
            self.result_dir = osp.join(osp.abspath(osp.dirname(__file__)), "..", "..")
        elif test_sdf == True and exp_name == 'genhand':
            self.result_dir = osp.join(osp.abspath(osp.dirname(__file__)), "..", "..", "results_sdf", self.robot_name, "mu"+str(mu))
        elif test_sdf == True and exp_name == 'nv':
            self.result_dir = osp.join(osp.abspath(osp.dirname(__file__)), "..", "..", "results_nv_sdf", self.robot_name, "mu"+str(mu))
        
        self.hand_pred_pose_dir = osp.join(osp.abspath(osp.dirname(__file__)), "..", "..", "CtcSDF_v2", "hmano_osdf", "hand_pose_results")
        self.obj_pred_pose_dir = osp.join(osp.abspath(osp.dirname(__file__)), "..", "..", "CtcSDF_v2", "hmano_osdf", "obj_pose_results")
        self.obj_pred_mesh_dir = osp.join(osp.abspath(osp.dirname(__file__)), "..", "..", "CtcSDF_v2", "hmano_osdf", "mesh")
        self.mesh_dir = osp.join(data_dir, 'mesh_data', 'mesh_obj')
        self.repeat = repeat
        self.load_assert = load_assert
        config_file = osp.join(data_dir, "dexycb_{}_s0.json".format(task))
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        if data_sample is not None:
            sample_dir = osp.join(data_dir, "test_sample_{}.json".format(data_sample))
            with open(sample_dir, 'r') as f:
                sample = json.load(f)
                all_indices = [np.array(i) for i in sample.values()]
                self.sample = np.concatenate(all_indices)
        else:
            self.sample = None
            
        if self.load_assert:
            self.mano_layer = ManoLayer(
                        ncomps = 45,
                        side = 'right',
                        mano_root = os.path.join(os.path.dirname(__file__),'..', '..', 'manopth','mano','models'),
                        use_pca=True,
                        flat_hand_mean=False
                        )
            self.faces = torch.LongTensor(np.load(osp.join(osp.dirname(osp.abspath(__file__)),'..', '..', 'CtcSDF','closed_fmano.npy')))            
        
    def __len__(self):
        return len(os.listdir(self.result_dir))

    def __getitem__(self, idx_):


        sample_id = idx_ // self.repeat
        res_id = idx_ % self.repeat
        if self.sample is not None:
            idx = self.sample[sample_id]
        else:
            idx = sample_id

        s0_id = self.config['images'][idx]['id']
        file_name = self.config['images'][idx]['file_name']
        sample = self.getdata[s0_id]
        ycb_id = sample['ycb_grasp_ind']
        label = np.load(sample['label_file'])
        pose_y = label['pose_y'][ycb_id]
        fx = sample['intrinsics']['fx']
        fy = sample['intrinsics']['fy']
        cx = sample['intrinsics']['ppx']
        cy = sample['intrinsics']['ppy']
        w = self.getdata.w
        h = self.getdata.h  

        if self.sdf_source:
            hand_trans = np.array(self.config['annotations'][idx]['hand_trans'], dtype = np.float32)
            obj_name = self.config["images"][idx]['file_name']+'.obj'
            obj_file = osp.normpath(osp.join(self.mesh_dir, obj_name))
            # print("dir check", obj_file)
            translation = - hand_trans
            quat = np.array([0, 0, 0, 1])    
            mat = np.eye(3)
            pose = np.vstack((np.hstack([mat, translation.reshape(3, 1)]), np.array([0, 0, 0, 1])))
        else:
            # print("using ycb source")
            # print("obj categroy", self.getdata.obj_file)
            obj_file = self.getdata.obj_file[sample['ycb_ids'][ycb_id]]
            # print("dir check", obj_file)
            pose = np.vstack((pose_y, np.array([[0, 0, 0, 1]], dtype=np.float32)))
            hand_trans = np.array(self.config['annotations'][idx]['hand_trans'], dtype = np.float32)
            translation = pose[:3, 3]- hand_trans
            pose[:3, 3] = pose[:3, 3]- hand_trans

            rotation = R.from_matrix(pose[:3, :3])
            quat = rotation.as_quat()
        
        if self.load_assert:
            color = cv2.imread(self.getdata[s0_id]['color_file'])
            pose_m = torch.from_numpy(label['pose_m'])
            bates = torch.tensor(sample['mano_betas'], dtype=torch.float32).unsqueeze(0)
            verts,_,_,_,_,_ = self.mano_layer( pose_m[:, 0:48],bates, pose_m[:, 48:51])
            #th_verts, th_jtr, th_full_pose,th_results_global,center_joint, root_trans
            # print("hand trans", pose_m[:, 48:51])
            verts = verts.view(778,3).numpy()
            pose_y_ = np.vstack((label['pose_y'][ycb_id],np.array([[0, 0, 0, 1]], dtype=np.float32)))
        
        obj_centre = np.array(self.config['annotations'][idx]['obj_center_3d'])
        obj_centre = obj_centre - hand_trans
        result_file = str(sample_id) + "_" + self.robot_name + "_" + str(res_id) + "_mu_" + str(self.mu) + ".json"
        print(osp.join(self.result_dir, result_file))
        # print("result file", result_file)
        try:
            r_quat, r_t, q_val, mat, q = self.opt_result_parser(osp.join(self.result_dir, result_file))
        except Exception as e:
            traceback.print_exc()
            print("Error in loading", result_file)
            r_quat = 0
            r_t = 0
            q_val = 0
            mat = 0
            q = 0
        if self.load_assert:
            return dict(file_name = file_name, color = color ,obj_file =obj_file, pose_y = pose_y_, hand_verts = verts, hand_faces = self.faces, obj_quat = quat, obj_trans = translation, obj_centre = obj_centre, rob_quat = r_quat, rob_trans = r_t, joint_val = q_val, rob_mat = mat, full_q = q, se3 = pose, fx = fx, fy = fy, cx = cx, cy = cy, w = w, h = h)
        else:
            return dict(obj_file =obj_file, obj_quat = quat, obj_trans = translation, obj_centre = obj_centre, rob_quat = r_quat, rob_trans = r_t, joint_val = q_val, rob_mat = mat, full_q = q, se3 = pose, fx = fx, fy = fy, cx = cx, cy = cy, w = w, h = h)
        
    def opt_result_parser(self, file_path: str) -> Dict:
        print(file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)
        # if data['distance_loss'] > 0.01:

        #     q = np.array(data['qr']).squeeze()
        # else:
        q = np.array(data['q']).squeeze()
        #global_rmat = rodrigues(torch.tensor(q[3:6], dtype=torch.float32)).view(3, 3).numpy()
        global_quat, mat = rodrigues(q[3:6])
        global_t = q[:3]
        if self.robot_name == "Robotiq":
            joint_val = q[6:] * np.array([-1, -1, 1, -1, 1, -1])
            # [-1, -1, 1, -1, 1, -1]
            # joint_val = q[6:] * np.array([1, 1, -1, -1, 1, -1])

        else:
            joint_val = q[6:]
        # se3 = np.vstack([np.hstack([mat, global_t.reshape(3, 1)]), np.array([0, 0, 0, 1])])
        
        return global_quat, global_t, joint_val, mat, q#, se3
