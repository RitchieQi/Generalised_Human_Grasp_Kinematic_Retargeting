import json
#import pickle
import os
import cv2
from manopth.manolayer import ManoLayer
from dex_ycb_toolkit.dex_ycb import DexYCBDataset
from torch.utils.data import Dataset
import torch
import numpy as np
import json
from dex_ycb_toolkit.factory import get_dataset
import open3d as o3d
import matplotlib.pyplot as plt
import trimesh
osp = os.path

class dexycb_testfullfeed(Dataset):  
    def __init__(self, data_dir = osp.join(osp.dirname(__file__), "data"),mano_root = os.path.join(os.path.dirname(__file__),'..', 'manopth','mano','models'), load_mesh = False, pc_sample = 1024, data_sample = None, scale_input=False, scale = None, precept = False):
        dexycb_dir = '/home/liyuan/DexYCB/'
        os.environ["DEX_YCB_DIR"] = dexycb_dir
        self.getdata = DexYCBDataset('s0', "test")
        self.scale_input = scale_input
        self.scale = scale
        config_file = osp.join(data_dir,"dexycb_test_s0.json")
        self.hand_pred_dir = osp.join(osp.dirname(__file__), '..', 'CtcSDF_v2', 'hmano_osdf', 'hand_pose_results')
        self.obj_pred_dir = osp.join(osp.dirname(__file__), '..', 'CtcSDF_v2', 'hmano_osdf', 'obj_pose_results')
        with open(config_file, 'r') as f:
            self.meta = json.load(f)
        self.data_sample = data_sample
        if data_sample is not None:
            sample_dir = osp.join(data_dir, "test_sample_{}.json".format(data_sample))
            with open(sample_dir, 'r') as f:
                sample = json.load(f)
                all_indices = [torch.tensor(i) for i in sample.values()]
                self.sample = torch.cat(all_indices)
            self.len = len(self.sample)
        else:
            self.len = len(self.meta['images'])
            self.sample = None

        self.load_mesh = load_mesh
        self.mano_root = mano_root
        self.precept = precept
        
        self.pred_obj_mesh_dir = osp.join(osp.dirname(__file__), '..', 'CtcSDF_v2', 'hmano_osdf', 'mesh')
        self.pred_hand_mesh_dir = osp.join(osp.dirname(__file__), '..', 'CtcSDF_v2', 'hmano_osdf', 'mesh_hand')
        self.mesh_dir = osp.join(data_dir, 'mesh_data', 'mesh_obj')
        self.point_dir = osp.join(data_dir, 'points_{}'.format(pc_sample))
        self.mano_layer = ManoLayer(
                    ncomps = 45,
                    side = 'right',
                    mano_root= self.mano_root,
                    use_pca=False,
                    flat_hand_mean=True
                    )
        self.mano_layer_pca = ManoLayer(
                    ncomps = 15,
                    side = 'right',
                    center_idx = 0,
                    mano_root= self.mano_root,
                    use_pca=True,
                    flat_hand_mean=True
                    )
        self.faces = torch.LongTensor(np.load(osp.join(osp.dirname(osp.abspath(__file__)),'closed_fmano.npy')))
    def __len__(self):
        return self.len

    def __getitem__(self, idx_):
        if self.sample is not None:
            idx = self.sample[idx_].item()
        else:
            idx = idx_
        s0_id = self.meta['images'][idx]['id']
        file_name = self.meta['images'][idx]['file_name']
        print("s0_id", self.meta['images'][idx]['file_name'])
        color = cv2.imread(self.getdata[s0_id]['color_file'])
        ycb_id = torch.tensor(self.meta['annotations'][idx]['ycb_id'], dtype = torch.long)
        dexycb_s0_id = torch.tensor(s0_id, dtype = torch.long)
        try:
            if self.precept:
                with open(osp.join(self.hand_pred_dir, file_name + '.json'), 'r') as f:
                    hand_pred_pose = json.load(f)
                cam_extr = torch.tensor(hand_pred_pose['cam_extr'], dtype = torch.float32)
                hand_mesh = trimesh.load_mesh(osp.join(self.pred_hand_mesh_dir, file_name + '_hand.ply'))
                obj_mesh = trimesh.load_mesh(osp.join(self.pred_obj_mesh_dir, file_name + '_obj.ply'))
                obj_centre = torch.tensor(obj_mesh.center_mass, dtype = torch.float32)
                hand_joint = torch.tensor(hand_pred_pose['joints'], dtype = torch.float32)
                hand_trans = 0
                global_trans = torch.tensor(hand_pred_pose['global_trans'], dtype = torch.float32)
                # mano_pose = torch.tensor(hand_pred_pose['pcas'], dtype = torch.float32).view(1, -1)
                # mano_shape = torch.tensor(hand_pred_pose['shape'], dtype = torch.float32).view(1, -1)
                # verts, th_jtr, th_full_pose, th_results_global, center_joint, th_trans = self.mano_layer_pca(th_pose_coeffs = mano_pose[:,0:18], th_betas = mano_shape, th_trans=torch.tensor([0.0,0.0,0.0]), root_palm=False)  
                # transform = torch.eye(4)
                # transform[:3, :3] = cam_extr
                # th_trans_ = (transform @ th_trans.squeeze(0).transpose(1,0)).transpose(1,0)
                # # print(th_trans_)
                # # new_trans = torch.eye(4)
                # # new_trans[:3, :3] = th_trans_[:3, :3]
                new_trans = global_trans.reshape(16,4,4)[0]
                new_trans[:3,3] = (cam_extr @ hand_joint[0].reshape(1,-1).transpose(1,0)).transpose(1,0)

                pose = torch.eye(4, dtype = torch.float32)
                return dict(color_img=color, s0 = dexycb_s0_id, file_name = file_name), dict(verts=torch.tensor(np.array(obj_mesh.vertices), dtype=torch.float32)-hand_trans, faces=torch.tensor(np.array(obj_mesh.faces)), pose=pose, centre=obj_centre, hand_trans = hand_trans), dict(verts=torch.tensor(np.array(hand_mesh.vertices, dtype=np.float32)), faces=self.faces, joint=hand_joint, transformation=new_trans.squeeze(0))
            else:

                hand_trans = torch.tensor(self.meta['annotations'][idx]['hand_trans'], dtype = torch.float32)
                hand_joint = torch.tensor(self.meta['annotations'][idx]['hand_joints_3d'], dtype = torch.float32)
                hand_joint = hand_joint - hand_trans
                obj_centre = torch.tensor(self.meta['annotations'][idx]['obj_center_3d'], dtype = torch.float32)
                mano_pose = torch.tensor(self.meta['annotations'][idx]['hand_poses'], dtype = torch.float32).view(1, -1)
                mano_shape = torch.tensor(self.meta['annotations'][idx]['hand_shapes'], dtype = torch.float32).view(1, -1)
                verts, th_jtr, th_full_pose, th_results_global, center_joint, th_trans = self.mano_layer(th_pose_coeffs = mano_pose[:,0:48], th_betas = mano_shape, th_trans=torch.tensor([0.0,0.0,0.0]), root_palm=False)  
                obj_centre = obj_centre - hand_trans


                mesh_name = self.meta['images'][idx]['file_name'] + '.obj' #already transformed
                mesh = trimesh.load_mesh(osp.join(self.mesh_dir, mesh_name))
                pose = torch.tensor(self.meta['annotations'][idx]['obj_transform'])
                return dict(color_img=color, ycb = ycb_id, s0 = dexycb_s0_id, file_name = file_name), dict(verts=torch.tensor(np.array(mesh.vertices))-hand_trans, faces=torch.tensor(np.array(mesh.faces)), pose=pose, centre=obj_centre, hand_trans = hand_trans), dict(verts=verts.squeeze(0), faces=self.faces, joint=hand_joint, transformation=th_trans.squeeze(0))

        except Exception as e:
            print(idx_, e)
            return self.__getitem__(idx_+1)


def write_json(data, filename):
    
    def convert_tensor_to_list(data):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.tolist()
        return data
    file_name = osp.join(osp.dirname(osp.abspath(__file__)), "data", filename+".json")
    with open(file_name, 'w') as f:
        json.dump(convert_tensor_to_list(data), f, indent=4)

def image_inspection(images):
    global im_id
    im_id = 0
    fig, ax = plt.subplots()
    im = images[im_id][:,:,::-1]
    im_display = ax.imshow(im)

    def update(event):
        global im_id
        if event.key == 'right':
            im_id = (im_id + 1) % len(images)
        elif event.key == 'left':
            im_id = (im_id - 1) % len(images)
        im = images[im_id][:,:,::-1]
        im_display.set_data(im)
        fig.canvas.draw()
    
    fig.canvas.mpl_connect('key_press_event', update)
    plt.show()




    
        
