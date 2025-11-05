import os
import cv2
import torch
import copy
import random
import numpy as np
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import os.path as osp
import json
import os
from dex_ycb_toolkit.dex_ycb import DexYCBDataset
from .utils import PerspectiveCamera
from kornia.geometry.conversions import rotation_matrix_to_angle_axis, angle_axis_to_rotation_matrix
import trimesh
from scipy.spatial import cKDTree as KDTree

class CtcSDF_dataset(Dataset):
    def __init__(self, start = None, end = None, data_dir = osp.join(osp.dirname(__file__),'data'), task='train', sdf_sample=2048, sdf_scale=6.2, clamp=0.1, input_image_size=(256, 256)):
        self.start = start
        self.end = end
        self.task = task
        self.data_dir = data_dir
        self.sdf_sample = sdf_sample
        self.clamp = clamp
        self.input_image_size = input_image_size
        config_file = osp.join(data_dir,"dexycb_{}_s0.json".format(task))
        with open(config_file, 'r') as f:
            self.json_data = json.load(f)
        self.images_dir = osp.join(data_dir, 'images', task)
        self.sdf_dir = osp.join(data_dir, "sdf_data")
        self.recon_scale= sdf_scale
        self.sdf_sample = sdf_sample
        os.environ["DEX_YCB_DIR"] = '/home/liyuan/DexYCB/'
        self.getdata = DexYCBDataset('s0', task)
        self.faces = torch.LongTensor(np.load(osp.join(osp.dirname(__file__),'data', 'closed_fmano.npy')))

        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    def __len__(self):
        if self.start is None:
            self.start = 0
        if self.end is None:
            self.len = len(self.json_data['images'])
        else:
            self.len = len(self.json_data['images'][self.start:self.end])
        return self.len
    
    def load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)  # type: ignore
        if not isinstance(img, np.ndarray):
            raise IOError("Fail to read %s" % path)  
         
        img = img[:,:,::-1].copy()
        img = img.astype(np.uint8)
        return img
             
    def __getitem__(self, idx):
        """
        data from DexYCB dataset
        """        
        if self.start is None:
            self.start = 0
        if self.end is None:
            self.end = len(self.json_data['images'])
        idx = idx + self.start 
        s0_id = self.json_data['images'][idx]['file_name']

        sample_data = copy.deepcopy(self.json_data['annotations'][idx])

        bbox = sample_data['bbox']
        try:
            hand_joints_3d = torch.tensor(sample_data['hand_joints_3d'], dtype = torch.float32)
        except:
            hand_joints_3d = torch.zeros((21, 3))

        try:
            hand_poses = torch.tensor(sample_data['hand_poses'], dtype = torch.float32)
        except:
            hand_poses = torch.zeros((48))

        try:
            hand_shapes = torch.tensor(sample_data['hand_shapes'], dtype = torch.float32)
        except:
            hand_shapes = torch.zeros((10))
        
        try:
            obj_center_3d = torch.tensor(sample_data['obj_center_3d'], dtype = torch.float32)
        except:
            obj_center_3d = torch.zeros(3)        
        try:
            obj_transform = torch.tensor(sample_data['obj_transform'], dtype = torch.float32)
        except:
            obj_transform = torch.zeros((4, 4))

        if self.task == 'train':
            sdf_name = osp.join(self.sdf_dir,"sdf_obj",self.json_data['images'][idx]['file_name'] + '.npz')
            sdf_scale = torch.tensor(self.json_data['annotations'][idx]['sdf_scale'], dtype = torch.float32)
            sdf_offset = torch.tensor(self.json_data['annotations'][idx]['sdf_offset'], dtype = torch.float32)
        
        img_data = self.json_data['images'][idx]['file_name']+ '.png'
        img_path = osp.join(self.images_dir, img_data)
        img = self.load_image(img_path)
        
        camera = PerspectiveCamera(sample_data['fx'], 
                                   sample_data['fy'], 
                                   sample_data['cx'], 
                                   sample_data['cy'])

        if self.task == 'train':
            trans, scale, rot, do_flip, color_scale = self.get_aug_config()
            rot_aug_mat = torch.from_numpy(np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0], [0, 0, 1]], dtype=np.float32))
        else:
            trans, scale, rot, do_flip, color_scale = [0, 0], 1, 0, False, [1.0, 1.0, 1.0]
        
        img, _ = self.generate_patch_image(img, [0, 0, self.input_image_size[1], self.input_image_size[0]], self.input_image_size, do_flip, scale, rot)

        for i in range(3):
            img[:, :, i] = np.clip(img[:, :, i] * color_scale[i], 0, 255)
            
        camera.update_virtual_camera_after_crop(bbox)
        camera.update_intrinsics_after_resize((bbox[-1], bbox[-2]), self.input_image_size)
        rot_cam_extr = torch.from_numpy(camera.extrinsics[:3, :3].T)
        
        if self.task == 'train':
            rot_aug_mat = rot_aug_mat @ rot_cam_extr
                        
        img = self.image_transform(img)
        cam_intr = torch.from_numpy(camera.intrinsics)
        cam_extr = torch.from_numpy(camera.extrinsics)
        
        if self.task == 'train':
            obj_samples, obj_labels = self.unpack_sdf(sdf_name, self.sdf_sample, hand=False, clamp=self.clamp, filter_dist=True)
            obj_samples[:, 0:3] = obj_samples[:, 0:3] / sdf_scale - sdf_offset
            obj_lablels = obj_labels.long() 
            obj_samples[:, 0:3] = torch.mm(rot_aug_mat, obj_samples[:, 0:3].transpose(1, 0)).transpose(1, 0)
            hand_joints_3d[:, 0:3] = torch.mm(rot_aug_mat, hand_joints_3d[:, 0:3].transpose(1, 0)).transpose(1, 0)
            hand_poses[:3] = rotation_matrix_to_angle_axis(rot_aug_mat @ angle_axis_to_rotation_matrix(hand_poses[:3].unsqueeze(0))).squeeze(0)
            obj_center_3d = torch.mm(rot_aug_mat, obj_center_3d.unsqueeze(1)).squeeze()
            trans_with_rot = torch.zeros((4, 4))
            trans_with_rot[:3, :3] = rot_aug_mat
            trans_with_rot[3, 3] = 1.
            obj_transform = torch.mm(trans_with_rot, obj_transform)
            obj_transform[:3, 3] = obj_transform[:3, 3] - hand_joints_3d[0]
            hand_center_3d = hand_joints_3d[0]
            obj_samples[:, 0:3] = (obj_samples[:, 0:3] - hand_center_3d) * self.recon_scale
            obj_samples[:, 3:] = obj_samples[:, 3:] / sdf_scale * self.recon_scale
            obj_samples[:, 0:5] = obj_samples[:, 0:5] / 2            
            input_iter = dict(img=img)
            label_iter = dict(obj_sdf=obj_samples, obj_labels=obj_labels, hand_joints_3d=hand_joints_3d, obj_center_3d=obj_center_3d)
            meta_iter = dict(cam_intr=cam_intr, cam_extr=cam_extr, id=s0_id, hand_center_3d=hand_center_3d, hand_poses=hand_poses, hand_shapes=hand_shapes, obj_transform=obj_transform)

            return input_iter, label_iter, meta_iter
        else:            
            hand_center_3d = torch.mm(rot_cam_extr, hand_joints_3d[:, 0:3].transpose(1, 0)).transpose(1, 0)[0]
            trans_with_rot = torch.zeros((4, 4))
            trans_with_rot[:3, :3] = rot_cam_extr
            trans_with_rot[3, 3] = 1.
            obj_transform = torch.mm(trans_with_rot, obj_transform)
            obj_transform[:3, 3] = obj_transform[:3, 3] - hand_center_3d
            input_iter = dict(img=img)
            meta_iter = dict(cam_intr=cam_intr, cam_extr=cam_extr, id=s0_id, hand_center_3d=hand_center_3d, hand_poses=hand_poses, hand_shapes=hand_shapes,  obj_transform=obj_transform)
            return input_iter, meta_iter

    def _evaluate(self, output_path, idx):
        if self.start is None:
            self.start = 0
        if self.end is None:
            self.end = len(self.json_data['images'])
        idx = idx + self.start 
        
        s0_id = self.json_data['images'][idx]['file_name']
        sample_data = copy.deepcopy(self.json_data['annotations'][idx])
        pred_mano_pose_path = osp.join(output_path, 'hand_pose_results', s0_id + '.json')
        with open(pred_mano_pose_path, 'r') as f:
            pred_hand_pose = json.load(f)
        cam_extr = np.array(pred_hand_pose['cam_extr'])
        try:
            pred_mano_joint =  (cam_extr @ np.array(pred_hand_pose['joints']).transpose(1, 0)).transpose(1, 0)
            mano_joint_err = np.mean(np.linalg.norm(pred_mano_joint - sample_data['hand_joints_3d'], axis=1)) * 1000.
        except:
            mano_joint_err = None

        pred_obj_pose_path = osp.join(output_path, 'obj_pose_results', s0_id + '.json')
        with open(pred_obj_pose_path, 'r') as f:
            pred_obj_pose = json.load(f)
        cam_extr = np.array(pred_obj_pose['cam_extr'])
        try:
            pred_obj_center = (cam_extr @ np.array(pred_obj_pose['center']).reshape((1, 3)).transpose(1, 0)).squeeze()
            obj_center_err = np.linalg.norm(pred_obj_center - sample_data['obj_center_3d']) * 1000.
        except:
            obj_center_err = None
        
        pred_obj_mesh_path = osp.join(output_path, 'mesh', s0_id + '_obj.ply')
        gt_obj_mesh_path = osp.join(self.data_dir, 'mesh_data', 'mesh_obj', s0_id + '.obj')
        try:
            pred_obj_mesh = trimesh.load(pred_obj_mesh_path, process=False)
            gt_obj_mesh = trimesh.load(gt_obj_mesh_path, process=False)

            pred_obj_points, _, *_ = trimesh.sample.sample_surface(pred_obj_mesh, 30000)
            gt_obj_points, _, *_ = trimesh.sample.sample_surface(gt_obj_mesh, 30000)
            pred_obj_points *= 100.
            gt_obj_points *= 100.            
            
            gen_points_kd_tree = KDTree(pred_obj_points)
            one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_obj_points)
            gt_to_gen_chamfer = np.mean(np.square(one_distances))
            
            gt_points_kd_tree = KDTree(gt_obj_points)
            two_distances, two_vertex_ids = gt_points_kd_tree.query(pred_obj_points)
            gen_to_gt_chamfer = np.mean(np.square(two_distances))
            chamfer_obj = gt_to_gen_chamfer + gen_to_gt_chamfer
            
            threshold = 0.5 # 5 mm
            precision_1 = np.mean(one_distances < threshold).astype(np.float32)
            precision_2 = np.mean(two_distances < threshold).astype(np.float32)
            fscore_obj_5 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)

            threshold = 1.0 # 10 mm
            precision_1 = np.mean(one_distances < threshold).astype(np.float32)
            precision_2 = np.mean(two_distances < threshold).astype(np.float32)
            fscore_obj_10 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)
        except:
            chamfer_obj = None
            fscore_obj_5 = None
            fscore_obj_10 = None
        
        pred_hand_mesh_path = osp.join(output_path, 'mesh_hand', s0_id + '_hand.ply')
        gt_hand_mesh_path = osp.join(self.data_dir, 'mesh_data', 'mesh_hand', s0_id + '.obj')
        try:
            pred_hand_mesh = trimesh.load(pred_hand_mesh_path, process=False)
            gt_hand_mesh = trimesh.load(gt_hand_mesh_path, process=False)

            pred_hand_points, _ = trimesh.sample.sample_surface(pred_hand_mesh, 30000)
            gt_hand_points, _ = trimesh.sample.sample_surface(gt_hand_mesh, 30000)
            pred_hand_points *= 100.
            gt_hand_points *= 100.            
            
            gen_points_kd_tree = KDTree(pred_hand_points)
            one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_hand_points)
            gt_to_gen_chamfer = np.mean(np.square(one_distances))
            
            gt_points_kd_tree = KDTree(gt_hand_points)
            two_distances, two_vertex_ids = gt_points_kd_tree.query(pred_hand_points)
            gen_to_gt_chamfer = np.mean(np.square(two_distances))
            chamfer_hand = gt_to_gen_chamfer + gen_to_gt_chamfer


            threshold = 0.1 # 1 mm
            precision_1 = np.mean(one_distances < threshold).astype(np.float32)
            precision_2 = np.mean(two_distances < threshold).astype(np.float32)
            fscore_hand_1 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)


            threshold = 0.5 # 5 mm
            precision_1 = np.mean(one_distances < threshold).astype(np.float32)
            precision_2 = np.mean(two_distances < threshold).astype(np.float32)
            fscore_hand_5 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)
        except:
            chamfer_hand = None
            fscore_hand_1 = None
            fscore_hand_5 = None
        error_dict = {}
        error_dict['id'] = s0_id
        error_dict['chamfer_obj'] = chamfer_obj
        error_dict['fscore_obj_5'] = fscore_obj_5
        error_dict['fscore_obj_10'] = fscore_obj_10
        error_dict['mano_joint'] = mano_joint_err
        error_dict['obj_center'] = obj_center_err
        error_dict['chamfer_hand'] = chamfer_hand
        error_dict['fscore_hand_1'] = fscore_hand_1
        error_dict['fscore_hand_5'] = fscore_hand_5
        return error_dict                            
     
    def unpack_sdf(self, data_path, subsample=None, hand=True, clamp=None, filter_dist=False):
            """
            @description: unpack sdf samples.
            ---------
            @param: sdf data path, num points, whether is hand, clamp dist, whether filter
            -------
            @Returns: points with sdf, part labels (only meaningful for hands)
            -------
            """

            def filter_invalid_sdf(tensor, lab_tensor, dist):
                keep = (torch.abs(tensor[:, 3]) < abs(dist)) & (torch.abs(tensor[:, 4]) < abs(dist))
                return tensor[keep, :], lab_tensor[keep, :]

            def remove_nans(tensor):
                tensor_nan = torch.isnan(tensor[:, 3])
                return tensor[~tensor_nan, :]

            npz = np.load(data_path)

            try:
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
                pos_sdf_other = torch.from_numpy(npz["pos_other"])
                neg_sdf_other = torch.from_numpy(npz["neg_other"])
                if hand:
                    lab_pos_tensor = torch.from_numpy(npz["lab_pos"])
                    lab_neg_tensor = torch.from_numpy(npz["lab_neg"])
                else:
                    lab_pos_tensor = torch.from_numpy(npz["lab_pos_other"])
                    lab_neg_tensor = torch.from_numpy(npz["lab_neg_other"])
            except Exception as e:
                print("fail to load {}, {}".format(data_path, e))

            if hand:
                pos_tensor = torch.cat([pos_tensor, pos_sdf_other], 1)
                neg_tensor = torch.cat([neg_tensor, neg_sdf_other], 1)
            else:
                xyz_pos = pos_tensor[:, :3]
                sdf_pos = pos_tensor[:, 3].unsqueeze(1)
                pos_tensor = torch.cat([xyz_pos, pos_sdf_other, sdf_pos], 1)

                xyz_neg = neg_tensor[:, :3]
                sdf_neg = neg_tensor[:, 3].unsqueeze(1)
                neg_tensor = torch.cat([xyz_neg, neg_sdf_other, sdf_neg], 1)

            # split the sample into half
            half = int(subsample / 2)

            if filter_dist:
                pos_tensor, lab_pos_tensor = filter_invalid_sdf(pos_tensor, lab_pos_tensor, 2.0)
                neg_tensor, lab_neg_tensor = filter_invalid_sdf(neg_tensor, lab_neg_tensor, 2.0)

            random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
            random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

            sample_pos = torch.index_select(pos_tensor, 0, random_pos)
            sample_neg = torch.index_select(neg_tensor, 0, random_neg)

            # label
            sample_pos_lab = torch.index_select(lab_pos_tensor, 0, random_pos)
            sample_neg_lab = torch.index_select(lab_neg_tensor, 0, random_neg)

            # hand part label
            # 0-finger corase, 1-finger fine, 2-contact, 3-sealed wrist
            hand_part_pos = sample_pos_lab[:, 0]
            hand_part_neg = sample_neg_lab[:, 0]
            samples = torch.cat([sample_pos, sample_neg], 0)
            labels = torch.cat([hand_part_pos, hand_part_neg], 0)

            if clamp:
                labels[samples[:, 3] < -clamp] = -1
                labels[samples[:, 3] > clamp] = -1

            if not hand:
                labels[:] = -1

            return samples, labels
    
    def get_aug_config(self):
        """
        @description: Modfied from https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/blob/master/data/dataset.py.
                      Set augmentation configs for different datasets.
        ---------
        @param: the name of the dataset.
        -------
        @Returns: parameters for different augmentations, including rotation, flip and color.
        -------
        """



        trans_factor = 0.10
        scale_factor = 0.0
        rot_factor = 45.
        enable_flip = False
        color_factor = 0.2


        trans = [np.random.uniform(-trans_factor, trans_factor), np.random.uniform(-trans_factor, trans_factor)]
        scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
        rot = np.clip(np.random.randn(), -2.0, 2.0) * rot_factor if random.random() <= 0.6 else 0

        if enable_flip:
            do_flip = random.random() <= 0.5
        else:
            do_flip = False

        c_up = 1.0 + color_factor
        c_low = 1.0 - color_factor
        color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]

        return trans, scale, rot, do_flip, color_scale
    def generate_patch_image(self, cvimg, bbox, input_shape, do_flip, scale, rot):
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

        if do_flip:
            img = img[:, ::-1, :]
            bb_c_x = img_width - bb_c_x - 1

        trans = self.gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, input_shape[1], input_shape[0], scale, rot, inv=False)
        img_patch = cv2.warpAffine(img, trans, (int(input_shape[1]), int(input_shape[0])), flags=cv2.INTER_LINEAR)
        new_trans = np.zeros((3, 3), dtype=np.float32)
        new_trans[:2, :] = trans
        new_trans[2, 2] = 1

        return img_patch, new_trans

    def gen_trans_from_patch_cv(self, c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
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
    
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = CtcSDF_dataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, data in enumerate(dataloader):
        print(i)
        print(data)
        break
    
    