import torch
import pytorch_kinematics as pk
import urdf_parser_py.urdf as u
from scipy.spatial.transform import Rotation as R
from plotly import graph_objects as go
import json
from pytorch_kinematics.urdf_parser_py.urdf import URDF as urdf
from manopth.rodrigues_layer import batch_rodrigues
import numpy as np
import os
import trimesh as tm
from optimisation_tmp.utils import *
from pytorch3d.structures import Meshes
from abc import ABC, abstractmethod
from scipy.optimize import linear_sum_assignment
from torch.nn.functional import relu
import optimisation_tmp.icp as icp

osp = os.path

class CtcRobot(ABC):
    """Base class for all robots

    """
    def __init__(self,
                 batch: int, 
                 device: str
                 ) -> None:
        self.device = device
        self.batch = batch
        self.contacts_num = None
        self.global_translation = None
        self.global_rotation = None
        self.current_status = None
        self.chain = None
        self.joint_lower = None
        self.joint_upper = None
        self.q = None
        self.meshV = {}
        self.meshF = {}
        self.collision_links = []
        self.robot_contact_points = {}
        self.robot_contact_normal = {}
        self.contact_links = []
        self.q_init = None
        self.joint_num = None
        self.key_link = [] #in the order of wrist, index, mid and so on
        self.key_link_ = []
        self.calibration_rotation = None
        self.calibration_translation = None
        self.scale = 1
        self.surface_points = {}
        self.surface_points_normal = {}
        self.dense_sample_link = []
        self.row_ind = None
        self.col_ind = None
        self.robot_model = None
        self.minuend =  None
        self.subtrahend = None 
    def forward_kinematics(self, q = None):
        if q is None:
            q = self.q

        if self.robot_model == 'Robotiq':
            full_q = q[:, -1:]*self.expend_
            q = torch.cat([q[:,:6], full_q], dim=-1)

        R_G = batch_rodrigues(q[:,3:6]).view(-1,3,3)
        T_G = q[:, :3].view(-1,3,1)
        R_f = torch.matmul(R_G, self.calibration_rotation)
        T_f = -torch.matmul(R_G, torch.matmul(self.calibration_rotation, self.calibration_translation)) + T_G

        self.global_translation = T_f.permute(0,2,1)
        self.global_rotation = R_f

        self.current_status = self.chain.forward_kinematics(q[:,6:])    

    def init_q(self, transformation = None):
        q = torch.zeros(self.batch, 3+3+self.joint_num, device=self.device)
        if transformation is None:
            q[:, 6:] = self.q_init
            self.q = q
        else:
            q[:, :3] = transformation[:, :3, 3]
            #q[:, 1] = q[:, 1] - 0.1
            #q[:, 0] = q[:, 0] - 0.1
            q[:, 3:6] = mat2axisang(transformation[:, :3, :3]).to(self.device)
            q[:, 6:] = self.q_init
            self.q = q
        self.q.requires_grad = True

    def get_mesh_updated(self, i = 0, q = None, color = 'lightblue', opacity = 0.5):
        data = []
        if q is not None:
            self.forward_kinematics(q)
        for ids, link in enumerate(self.meshV):
            trans_m = self.current_status[link].get_matrix()
            trans_m = trans_m[0].detach().cpu().numpy()
            v = self.meshV[link]
            transed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transed_v = np.matmul(trans_m, transed_v.T).T[..., :3]
            transed_v = np.matmul(self.global_rotation[0].detach().cpu().numpy(),
                                    transed_v.T).T + np.expand_dims(
                                    self.global_translation[0].detach().cpu().numpy(), 0)
            transed_v = transed_v.squeeze()
            f = self.meshF[link]
            data.append(
                go.Mesh3d(x=transed_v[:, 0], y=transed_v[:, 1], z=transed_v[:, 2], i=f[:, 0], j=f[:, 1],
                          k=f[:, 2], color=color, opacity=opacity))
            # data.append(
            #     go.Scatter3d(x= (transed_v.mean(0))[None,0] , y=(transed_v.mean(0))[None,1], z=(transed_v.mean(0))[None,2], mode='markers+text', text=link,
            #         textposition='top center', marker=dict(size=2, color='red'), name='vertices'))
        return data     

    def get_surface_points_updated(self, q = None):
        if q is not None:
            self.forward_kinematics(q)
        points = []
        for name in self.collision_links:
            trans_matrix = self.current_status[name].get_matrix()
            points.append(torch.matmul(trans_matrix, self.surface_points[name].transpose(1,2)).transpose(1,2)[..., :3])
        surface_points = torch.cat(points, dim=1)
        #print(surface_points.size(),"surface_points")
        surface_points = torch.matmul(self.global_rotation, surface_points.transpose(1,2)).transpose(1,2) + self.global_translation
        #print(surface_points.size(),"surface_points", self.global_translation.size())
        return surface_points.squeeze()*self.scale

    def get_fingertip_surface_points_updated(self, q = None):
        if q is not None:
            self.forward_kinematics(q)
        points = []
        for fingertip in self.dense_sample_link:
            trans_matrix = self.current_status[fingertip].get_matrix()
            points_robot = torch.matmul(trans_matrix, self.surface_points[fingertip].transpose(1,2)).transpose(1,2)[..., :3]
            points_global = torch.matmul(self.global_rotation, points_robot.transpose(1,2)).transpose(1,2) + self.global_translation
            points.append(points_global*self.scale)
        return points
    
    def get_surface_points_normal_updated(self, q = None):
        if q is not None:
            self.forward_kinematics(q)
        points = []
        normals = []
        for name in self.collision_links:
            trans_matrix = self.current_status[name].get_matrix()
            points.append(torch.matmul(trans_matrix, self.surface_points[name].transpose(1,2)).transpose(1,2)[..., :3])
            normals.append(torch.matmul(trans_matrix, self.surface_points_normal[name].transpose(1,2)).transpose(1,2)[..., :3])
        surface_points = torch.cat(points, dim=1)
        surface_normals = torch.cat(normals, dim=1)
        surface_points = torch.matmul(self.global_rotation, surface_points.transpose(1,2)).transpose(1,2) + self.global_translation
        surface_normals = torch.matmul(self.global_rotation, surface_normals.transpose(1,2)).transpose(1,2)
        return surface_points.squeeze()*self.scale, surface_normals.squeeze()
    
    def get_finger_surface_points_normal_updated(self, q = None):
        if q is not None:
            self.forward_kinematics(q)
        points = []
        normals = []
        for finger in self.self_collision_links:
            finger_points = []
            finger_normals = []
            for name in self.self_collision_links[finger]:
                trans_matrix = self.current_status[name].get_matrix()
                finger_points.append(torch.matmul(trans_matrix, self.surface_points[name].transpose(1,2)).transpose(1,2)[..., :3])
                finger_normals.append(torch.matmul(trans_matrix, self.surface_points_normal[name].transpose(1,2)).transpose(1,2)[..., :3])
            finger_points = torch.cat(finger_points, dim=1)
            finger_normals = torch.cat(finger_normals, dim=1)
            finger_points = torch.matmul(self.global_rotation, finger_points.transpose(1,2)).transpose(1,2) + self.global_translation
            finger_normals = torch.matmul(self.global_rotation, finger_normals.transpose(1,2)).transpose(1,2)
            points.append(finger_points.squeeze()*self.scale)
            normals.append(finger_normals.squeeze())
        return points, normals
    
    def get_contact_points_normal_updated(self, q = None):
        if q is not None:
            self.forward_kinematics(q)
        points = []
        normals = []
        for name in self.contact_links:
            trans_matrix = self.current_status[name].get_matrix()
            points_ = self.robot_contact_points[name]
            normals_ = self.robot_contact_normal[name]
            points.append(torch.matmul(trans_matrix, points_.transpose(1,2)).transpose(1,2)[..., :3])
            normals.append(torch.matmul(trans_matrix[..., :3, :3], normals_.transpose(1,2)).transpose(1,2))
            #print("point & normal",points[-1].size(), normals[-1].size())
        J = len(points)
        contact_points = torch.stack(points, dim=1).view(1, -1, 3)#.mean(2)
        contact_normals = torch.stack(normals, dim=1).view(1, -1, 3)#.squeeze(2)
        #print(contact_points.size(), contact_normals.size())
        contact_points = torch.matmul(self.global_rotation, contact_points.transpose(1,2)).transpose(1,2) + self.global_translation
        contact_normals = torch.matmul(self.global_rotation, contact_normals.transpose(1,2)).transpose(1,2)
        return (contact_points.squeeze()*self.scale).view(1, J, -1, 3), contact_normals.squeeze().view(1, J, -1, 3)
        
    def robot_vertices(self, q = None):
        verts = []
        if q is not None:
            self.forward_kinematics(q)
        for ids, link in enumerate(self.collision_links):
            transm = self.current_status[link].get_matrix()
            transm = transm[0].detach().cpu().numpy()
            v = self.meshV[link]
            transedv = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transedv = np.matmul(transm, transedv.T).T[..., :3]
            transedv = np.matmul(self.global_rotation[0].detach().cpu().numpy(),
                                    transedv.T).T + np.expand_dims(
                                    self.global_translation[0].detach().cpu().numpy(), 0)
            transedv = torch.tensor(transedv).to(self.device)
            verts.append(transedv)
        return torch.cat(verts, dim=1).to(self.device)
    
    def robot_vertices_v2(self, q = None):
        verts = []
        if q is not None:
            self.forward_kinematics(q)
        for ids, link in enumerate(self.collision_links):
            transm = self.current_status[link].get_matrix()
            transm = transm[0]
            v = torch.tensor(self.meshV[link],dtype=torch.float).to(self.device)
            ones = torch.ones([len(v), 1], device=self.device)
            transedv = torch.cat([v, ones], dim=-1)
            transedv = torch.matmul(transm, transedv.T).T[..., :3]
            global_rot = self.global_rotation[0]
            global_trans = self.global_translation[0]
            transedv = torch.matmul(global_rot, transedv.T).T + global_trans
            verts.append(transedv)
        return torch.cat(verts, dim=0).to(self.device)

    def get_configurations(self, rot_rep = 'axisang'):
        if rot_rep == 'axisang':
            return dict(translate=self.q[:, :3], rotate=self.q[:, 3:6], joint=self.q[:, 6:])
        elif rot_rep == 'mat': # axisang to quat
            return dict(translate=self.q[:, :3], rotate=batch_rodrigues(self.q[:, 3:6]).view(-1,3,3), joint=self.q[:, 6:])
        elif rot_rep == 'quat':
            return dict(translate=self.q[:, :3], rotate=(axis2quat(self.q[:, 3:6])), joint=self.q[:, 6:])

    def get_keypoints(self, q = None, subset = None):
        if q is not None:
            self.forward_kinematics(q)
        if subset is not None:
            keypoints = []
            link_list = self.key_link_[:subset]
            for name in link_list:
                # print(name)
                pos, quat = quat_pos_from_transform3d(self.current_status[name])
                keypoints.append(pos)
        else:
            keypoints = []
            for name in self.key_link:
                pos, quat = quat_pos_from_transform3d(self.current_status[name])
                keypoints.append(pos)
        keypoints = torch.stack(keypoints, dim=1).to(self.device)
        keypoints = torch.matmul(self.global_rotation, keypoints.transpose(1,2)).transpose(1,2) + self.global_translation.unsqueeze(1)
        return keypoints

    def kinematics_distance(self, target, guide, q):
        robot_pose = self.get_keypoints(q).squeeze()
        target_num = target.size(1)
        finger_0 = robot_pose[-1]
        target_0 = target[:,0,:]
        
        robot_tips_utilized = robot_pose[1:target_num,:]
        target_tips_utilized = target[:,1:,:]
        #print("size",robot_tips_utilized.size(), target_tips_utilized.size())       
        distance_0 = torch.cdist(finger_0.view(-1,3), target_0.view(-1,3))
   
        #robot_tips_utilized = torch.cat((robot_pose[:,1:target_num,:], robot_pose[:,-1,:].unsqueeze(1)), dim=1)
        dist = torch.cdist(robot_tips_utilized.view(-1,3), target_tips_utilized.view(-1,3))
        row_ind, col_ind = linear_sum_assignment(dist.cpu().detach().numpy())
        #print(row_ind, col_ind, dist)
        #row_ind, col_ind = linear_sum_assignment(dist.cpu().detach().numpy())
        #distance_loss = (torch.sum(dist[self.row_ind, self.col_ind])/robot_tips_utilized.size(0)) #+ distance_0
        distance_loss = (torch.sum(dist[row_ind, col_ind]) + distance_0)/target_num
        joint_limit = relu(self.q[:, 6:] - self.joint_upper) + relu(self.joint_lower - self.q[:, 6:])
        #guide_distance = torch.norm(guide[:,0,:]-robot_pose[0,:].unsqueeze(0), dim=-1).mean()
        regularization = torch.norm(q[:, 6:], dim=-1).mean()
        return dict(distance_loss=distance_loss, joint_limit=joint_limit.sum(), distance_0=distance_0, loss=distance_loss + 10*joint_limit.sum() + 5e-4*regularization)
    
    def kinematics_distance_all(self, target, q, test = True, hand_joints = None):
        target_num = target.size(1)
        robot_pose = self.get_keypoints(q, subset = target_num).squeeze(0)

        target_tips_utilized = target
        # print("target_tips_utilized", target_tips_utilized.size())
        # print("target_num", target_num)
        #print("size",robot_tips_utilized.size(), target_tips_utilized.size())       
   
        # robot_tips_utilized = torch.cat((robot_pose[:,-1,:].unsqueeze(1), robot_pose[:,1:target_num,:]), dim=1)
        robot_tips_utilized = robot_pose
        if test and hand_joints is not None:
            dist = torch.cdist(robot_tips_utilized.view(-1,3), target_tips_utilized.view(-1,3))
            if self.row_ind is None:
                print(self.row_ind, self.col_ind)
                hand_joints_utilized = hand_joints.view(-1,3)[:target_num,:]
                h_normed = (hand_joints_utilized.view(-1,3) - hand_joints_utilized.view(-1,3).mean(0))/hand_joints_utilized.view(-1,3).norm(dim=-1).max()
                t_normed = (target_tips_utilized.view(-1,3) - target_tips_utilized.view(-1,3).mean(0))/target_tips_utilized.view(-1,3).norm(dim=-1).max()
                T,_,_,_ = icp.icp(h_normed.cpu().detach().numpy(), t_normed.cpu().detach().numpy(), max_iterations=100, tolerance=1e-6)
                T = torch.tensor(T, dtype=torch.float32).to(self.device)
                r_normed_transformed = (T[:3,:3] @ h_normed.T + T[:3,3].reshape(-1,1)).T
                # self.calibration_rotation = T[:3,:3] @ self.calibration_rotation
                dist_norm = torch.cdist(r_normed_transformed, t_normed)            
                self.row_ind, self.col_ind = linear_sum_assignment(dist_norm.cpu().detach().numpy())
            distance_loss = (torch.sum(dist[self.row_ind, self.col_ind])/robot_tips_utilized.size(0)) #+ distance_0
        elif test and hand_joints is None:
            raise NotImplementedError("hand_joints is None, please provide hand_joints")
        else:
            dist = torch.cdist(robot_tips_utilized.view(-1,3), target_tips_utilized.view(-1,3))
            
            r_normed = (robot_tips_utilized.view(-1,3) - robot_tips_utilized.view(-1,3).mean(0))/robot_tips_utilized.view(-1,3).norm(dim=-1).max()
            t_normed = (target_tips_utilized.view(-1,3) - target_tips_utilized.view(-1,3).mean(0))/target_tips_utilized.view(-1,3).norm(dim=-1).max()
            T,_,_,_ = icp.icp(r_normed.cpu().detach().numpy(), t_normed.cpu().detach().numpy(), max_iterations=100, tolerance=1e-6)
            T = torch.tensor(T, dtype=torch.float32).to(self.device)
            r_normed_transformed = (T[:3,:3] @ r_normed.T + T[:3,3].reshape(-1,1)).T
            dist_norm = torch.cdist(r_normed_transformed, t_normed)            
            self.row_ind, self.col_ind = linear_sum_assignment(dist_norm.cpu().detach().numpy())
            distance_loss = (torch.sum(dist[self.row_ind, self.col_ind])/robot_tips_utilized.size(0)) #+ distance_0
        
        joint_limit = relu(self.q[:, 6:] - self.joint_upper) + relu(self.joint_lower - self.q[:, 6:])
        regularization = torch.norm(q[:, 6:]-self.q_init, dim=-1).mean()
        return dict(distance_loss=distance_loss, joint_limit=joint_limit.sum(), loss=50*distance_loss + 100*joint_limit.sum() + 5e-4*regularization) #5e-4

    def kinematics_distance_wnormal(self, target, q):
        contact_point, contact_normal = self.get_contact_points_normal_updated(q) #B, J, N, 3      
        robot_pose = self.get_keypoints(q).squeeze(0) #B, J, 3
        target_contact_point, target_contact_normal = target #B, M, 3
        #print("size_1",contact_point.size(), target_contact_point.size())
        _,_,point_num,_ = contact_point.size()
        
        target_num = target_contact_point.size(1)
        robot_tips_utilized = torch.cat((robot_pose[:,1:target_num,:], robot_pose[:,-1,:].unsqueeze(1)), dim=1)
        contact_point_utilized = torch.cat((contact_point[:,1:target_num,:,:], contact_point[:,-1,:,:].unsqueeze(1)), dim=1) #B, M, N, 3
        contact_normal_utilized = torch.cat((contact_normal[:,1:target_num,:,:], contact_normal[:,-1,:,:].unsqueeze(1)), dim=1) #B, M, N, 3
        #print("size_2",robot_tips_utilized.size(), target_contact_point.size())
        dist = torch.cdist(robot_tips_utilized.view(-1,3), target_contact_point.view(-1,3))
        row_ind, col_ind = linear_sum_assignment(dist.cpu().detach().numpy())
        
        target_point_extend = target_contact_point.unsqueeze(2).repeat(1,1,point_num,1) #B, M, N, 3
        target_normal_extend = target_contact_normal.unsqueeze(2).repeat(1,1,point_num,1)
        #print("size_3",target_point_extend.size(), target_normal_extend.size())
        #select the corresponding target points
        contact_point_utilized = contact_point_utilized[:,col_ind,:,:] #B, M, N, 3
        contact_normal_utilized = contact_normal_utilized[:,col_ind,:,:] #B, M, N, 3
        #print("size_4",contact_point_utilized.size(), contact_normal_utilized.size())
        position_distance = torch.norm(contact_point_utilized - target_point_extend, dim=-1) #B, M, N
        normal_distance = torch.sum(contact_normal_utilized * target_normal_extend, dim=-1) #B, M, N
        #print("size_5",position_distance.size(), normal_distance.size())
        min_val, min_idx = torch.min(1*position_distance + 0.0*(1+normal_distance), dim=-1)
        contact_loss = min_val.mean()
        #min_idx = torch.argmin(0.5*position_distance + 0.5*normal_distance, dim=-1)
        #print("contact_loss",contact_loss, min_idx)
        joint_limit = relu(self.q[:, 6:] - self.joint_upper) + relu(self.joint_lower - self.q[:, 6:])
        regularization = torch.norm(q[:, 6:]-self.q_init, dim=-1).mean()
        
        return dict(distance_loss=position_distance.mean(), joint_limit=joint_limit.sum(), contact_loss = contact_loss, loss=contact_loss + 100*joint_limit.sum() + 5e-4*regularization)
        
    def kinematics_distance_aligned(self, target, q):
        contact_point, contact_normal = self.get_contact_points_normal_updated(q)
        target_contact_point, target_contact_normal = target
        pass
              
    def kinematics_distance_nv(self, target, q):
        """
        the method to optimize the robot posture using the NVD/UW methid
        target: (7, 3) np.ndarray, [wrist, index, middle, ring, pinky, thumb, object]
        wrist: target[0, :]  [wrist]
        fingertips: target[1:5, :] [index, middle, ring, pinky, thumb]
        thumb: target[5, :] [thumb]
        obj_c: target[6, :] [object]
        
        example: robot_kp_num = 5  # the number of keypoints of the robot hand, including the wrist
                 robot_finger_num = robot_kp_num - 1 # the number of the fingertips
                 
                 fingertip_subset = target[1:robot_finger_num -1 +1, :] [index, middle, ring]
                 fingertip_all = torch.cat((fingertip_subset, target[5, :].unsqueeze(0)), dim=0) [index, middle, ring, thumb]
                 
        q: robot joint angles
        robot_keypoints: [wrist, index, middle, ring, pinky, thumb]
        the method based the distance between the fingertips and the distance between the fingertips and the object
        The distance between the fingertips is calculated by the distance between the 2-combinations of the fingertips and wrist
        The distance between the fingertips and the object is calculated by the distance between the fingertips and the object
        """
        # the three povital points: wrist, thumb, object
        # the keypoints of the robot hand
        robot_pose = self.get_keypoints(q).squeeze(0)
        human_pose_ = target[:-1, :].unsqueeze(0) #[wrist, index, middle, ring, pinky, thumb]
        obj_c = target[-1, :].unsqueeze(0) #[object]
        
        robot_minuend = robot_pose[:, self.minuend, :]
        robot_subtrahend = robot_pose[:, self.subtrahend, :]
        robot_tips = robot_pose[:, 1:, :] # [index, ..., thumb]
        num_finger = robot_tips.size(1)
        
        human_pose = torch.cat((human_pose_[:, :num_finger, :], human_pose_[:, -1, :].unsqueeze(1)), dim=1) 
        human_minuend = human_pose[:, self.minuend, :] 
        human_subtrahend = human_pose[:, self.subtrahend, :]
        human_tips = human_pose[:, 1:, :]
        
        robot_vector = robot_minuend - robot_subtrahend
        human_vector = human_minuend - human_subtrahend
        dist_0 = torch.norm(robot_vector - human_vector, dim=2).pow(2).mean()
        
        robot_object = robot_tips - obj_c
        human_object = human_tips - obj_c
        dist_1 = torch.norm(robot_object - human_object, dim=2).pow(2).mean()
        
        
        #This guide loss is equivalent to dist_1
        #guide_loss = torch.norm(robot_tips - human_tips, dim=2).pow(2).mean()

        
        joint_limit = relu(self.q[:, 6:] - self.joint_upper) + relu(self.joint_lower - self.q[:, 6:])
        regularization = torch.norm(q[:, 6:]-torch.zeros_like(q[:, 6:]), dim=-1).mean()
        
        loss = dist_0 + dist_1 #+ guide_loss 
        #dist1 works alone
        #dist0 doesn't work alone UPDATE: dist0 works alone
        
        return dict(distance_loss=loss, finger_loss = dist_0, object_loss = dist_1, joint_limit=joint_limit.sum(), loss=20*loss + 100*joint_limit.sum() + 5e-4*regularization)

          
class Shadow(CtcRobot):
    def __init__(self, batch, device):
        super(Shadow, self).__init__(batch, device)
        urdf_path = osp.join(osp.dirname(__file__), '..', 'model', 'urdf', 'sr_description','urdf','shadow_hand.urdf')
        mesh_path = osp.join(osp.dirname(__file__), '..', 'model', 'urdf', 'sr_description','meshes', 'shadow_hand','hand')
        self.mesh_path = mesh_path
        self.robot_model = 'Shadow'
        self.minimal_dist = 0.005
        self.contacts_num = 5
        self.joint_num = 24
        self.key_link = ['rh_wrist','rh_fftip','rh_mftip','rh_rftip','rh_lftip','rh_thtip']
        self.key_link_ = ['rh_thtip', 'rh_fftip','rh_mftip','rh_rftip','rh_lftip','rh_wrist']

        # self.collision_links = ['rh_palm','rh_ffproximal','rh_ffmiddle','rh_ffdistal', 'rh_mfproximal',
        #                         'rh_mfmiddle','rh_mfdistal','rh_rfproximal','rh_rfmiddle','rh_rfdistal','rh_lfmetacarpal',
        #                         'rh_lfproximal','rh_lfmiddle','rh_lfdistal','rh_thbase','rh_thproximal','rh_thhub',
        #                         'rh_thmiddle','rh_thdistal']
        self.collision_links = ['rh_palm',
                                'rh_ffproximal','rh_ffmiddle','rh_ffdistal', 
                                'rh_mfproximal','rh_mfmiddle','rh_mfdistal',
                                'rh_rfproximal','rh_rfmiddle','rh_rfdistal',
                                'rh_lfmetacarpal','rh_lfproximal','rh_lfmiddle','rh_lfdistal',
                                'rh_thbase','rh_thproximal','rh_thhub','rh_thmiddle','rh_thdistal']
        self.contact_links = ['rh_ffdistal','rh_mfdistal','rh_rfdistal','rh_lfdistal','rh_thdistal']
        # self.self_collision_links = {'rh_ffdistal': ['rh_ffproximal','rh_ffmiddle','rh_ffdistal'],
        #                              'rh_mfdistal': ['rh_mfproximal','rh_mfmiddle','rh_mfdistal'],
        #                              'rh_rfdistal': ['rh_rfproximal','rh_rfmiddle','rh_rfdistal'],
        #                              'rh_lfdistal': ['rh_lfmetacarpal','rh_lfproximal','rh_lfmiddle','rh_lfdistal'],
        #                              'rh_thdistal': ['rh_thbase','rh_thproximal','rh_thhub','rh_thmiddle','rh_thdistal']}
        self.dense_sample_link = ['rh_ffdistal','rh_mfdistal','rh_rfdistal','rh_lfdistal','rh_thdistal']
        self.self_collision_links = {'rh_ffdistal': ['rh_ffmiddle'],
                                     'rh_mfdistal': ['rh_mfmiddle'],
                                     'rh_rfdistal': ['rh_rfmiddle'],
                                     'rh_lfdistal': ['rh_lfmiddle'],
                                     'rh_thdistal': ['rh_thmiddle']}
        #self.collision_links = ['rh_ffdistal','rh_mfdistal','rh_rfdistal','rh_lfdistal','rh_thdistal']
        # self.self_collision_links = {'rh_ffdistal': ['rh_ffmiddle','rh_ffdistal'],
        #                              'rh_mfdistal': ['rh_mfmiddle','rh_mfdistal'],
        #                              'rh_rfdistal': ['rh_rfmiddle','rh_rfdistal'],
        #                              'rh_lfdistal': ['rh_lfmiddle','rh_lfdistal'],
        #                              'rh_thdistal': ['rh_thmiddle','rh_thdistal']}
        self.contact_points_file = json.load(open(osp.join(osp.dirname(__file__), '..', 'model', 'urdf', 'contact_shadowhand.json')))
        self.calibration_rotation = batch_rodrigues(torch.tensor([0,-(torch.pi/2),0]).view(1,3)).view(1,3,3).to(self.device)
        # self.calibration_translation = torch.tensor([0.0000, -0.0100, 0.2130]).view(1,3,1).to(self.device)
        self.calibration_translation = torch.tensor([0.0000, -0.0100, 0.2130]).view(1,3,1).to(self.device)

        self.device = device
        self.chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(dtype=torch.float32, device=self.device)
        print(self.chain.get_joint_parameter_names())

        self.minuend =    torch.tensor([1,2,3,4,5,  1,2,3,4,  2,3,4,  2,4], dtype = torch.long, device=self.device)
        self.subtrahend = torch.tensor([0,0,0,0,0,  5,5,5,5,  1,1,1,  3,3], dtype = torch.long ,device=self.device)      
        self.joint_lower = torch.tensor([[-0.524, -0.698, -0.349, 0.0, 0.0, 0.0, -0.349, 0.0, 0.0, 0.0, -0.349, 0.0, 0.0, 0.0, 0.0, 
                                          -0.349, 0.0, 0.0, 0.0, -1.047, 0.0, -0.209, -0.698, 0.0]]
                                         ,dtype=torch.float32, device=self.device).repeat([self.batch, 1])
        self.joint_upper = torch.tensor([[0.175, 0.489, 0.349, 1.571, 1.571, 1.571, 0.349, 1.571, 1.571, 1.571, 0.349, 1.571, 1.571, 1.571, 0.785, 
                                          0.349, 1.571, 1.571, 1.571, 1.047, 1.222, 0.209, 0.698, 1.571]],dtype=torch.float32, device=self.device).repeat([self.batch, 1])
        # self.q_init = torch.tensor([0., 0., -0.349, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -0.349, 0., 0., 0., -1.047, 1.222, 0., 0., 0.],dtype=torch.float32, device=self.device).repeat([self.batch, 1])
        self.q_init = torch.zeros(24, dtype=torch.float32, device=self.device).repeat([self.batch, 1])
        for frame_name in self.chain.get_frame_names(exclude_fixed=False):
            frame = self.chain.find_frame(frame_name)
            for link_vis in frame.link.visuals:
                if link_vis.geom_type == "mesh":
                    filename = link_vis.geom_param[0].split("/")[-1]
                    mesh = tm.load(osp.join(mesh_path, filename), force='mesh', process=False)
                elif link_vis.geom_type == "box":
                    mesh = tm.primitives.Box(extents=link_vis.geom_param)
                elif link_vis.geom_type == None:
                    continue
                else:
                    #print(link_vis)
                    raise NotImplementedError
                try:
                    scale = np.array(link_vis.geom_param[1]).reshape([1,3])
                except:
                    #print(frame.link.name)
                    scale = np.array([[1,1,1]])
                try:
                    offset = link_vis.offset.get_matrix()
                    pos = offset[:,:3,3].numpy()
                    rot = offset[:,:3,:3].squeeze().numpy()
                except:
                    offset = pk.Transform3d().get_matrix()
                    pos = offset[:,:3,3].numpy()
                    rot = offset[:,:3,:3].squeeze().numpy()
                # print(frame.link.name, filename)
                self.meshV[frame.link.name] = np.array(mesh.vertices) * scale
                self.meshV[frame.link.name] = self.meshV[frame.link.name][:, [0, 2 , 1]]
                self.meshV[frame.link.name][:, 2] *= -1
                self.meshV[frame.link.name] = np.matmul(rot, self.meshV[frame.link.name].T).T + pos
                self.meshF[frame.link.name] = np.array(mesh.faces)
                
                if frame.link.name == 'rh_palm':
                    pts, pts_face_index = tm.sample.sample_surface(mesh=mesh, count=512)
                elif frame.link.name in self.dense_sample_link:
                    pts, pts_face_index = tm.sample.sample_surface(mesh=mesh, count=128)
                else:
                    pts, pts_face_index = tm.sample.sample_surface(mesh=mesh, count=32)
                pts_normal = np.array([mesh.face_normals[i] for i in pts_face_index], dtype=float)
                pts *= scale
                pts = np.matmul(rot, pts.T).T + pos
                pts_normal = np.matmul(rot, pts_normal.T).T
                pts = np.concatenate([pts, np.ones([len(pts), 1])], axis=-1)
                pts_normal = np.concatenate([pts_normal, np.ones([len(pts_normal), 1])], axis=-1)
                self.surface_points[frame.link.name] = torch.from_numpy(pts).to(self.device).float().unsqueeze(0).repeat(self.batch, 1, 1)
                self.surface_points_normal[frame.link.name] = torch.from_numpy(pts_normal).to(self.device).float().unsqueeze(0).repeat(self.batch, 1, 1)
                
                if frame.link.name in self.contact_links:
                    contact_pack = np.array(self.contact_points_file[frame.link.name])
                    cont_pts = contact_pack[0]*scale
                    cont_pts = np.matmul(rot, cont_pts.T).T + pos
                    cont_pts = torch.cat([torch.from_numpy(cont_pts).to(self.device).float(), torch.ones(cont_pts.shape[0],1).to(self.device).float()], dim=-1)
                    self.robot_contact_points[frame.link.name] = cont_pts.unsqueeze(0).repeat(self.batch, 1, 1)
                    cont_nls = contact_pack[1]
                    cont_nls = np.matmul(rot, cont_nls.T).T
                    self.robot_contact_normal[frame.link.name] = torch.from_numpy(cont_nls).to(self.device).float().unsqueeze(0).repeat(self.batch, 1, 1)
                #contact points
                # if frame.link.name in self.contact_points_file:
                #     contact_pack = np.array(self.contact_points_file[frame.link.name]) #2, 16, 3

                    
                #     contact_points = mesh.vertices[contact_points_i] * scale
                #     contact_points = np.matmul(rot, contact_points.T).T + pos
                #     contact_points = torch.cat([torch.from_numpy(contact_points).to(self.device).float(), torch.ones([4,1]).to(self.device).float()], dim=-1)
                #     self.robot_contact_points[frame.link.name] = contact_points.unsqueeze(0).repeat(self.batch, 1, 1)
                #     v1 = contact_points[1, :3] - contact_points[0, :3]
                #     v2 = contact_points[2, :3] - contact_points[0, :3]
                #     v1 = v1 / torch.norm(v1)
                #     v2 = v2 / torch.norm(v2)
                #     normal = torch.cross(v1, v2).view(1,3)
                #     self.robot_contact_normal[frame.link.name] = normal.unsqueeze(0).repeat(self.batch, 1, 1)


class Allegro(CtcRobot):
    def __init__(self, batch, device):
        super(Allegro, self).__init__(batch, device)
        urdf_path = osp.join(osp.dirname(__file__), '..', 'model', 'urdf', 'allegro_hand','allegro_hand_description_right.urdf')
        mesh_path = osp.join(osp.dirname(__file__), '..', 'model', 'urdf', 'allegro_hand','meshes')
        self.mesh_path = mesh_path

        self.robot_model = 'Allegro'
        self.contacts_num = 4
        self.joint_num = 16
        self.minimal_dist = 0.01
        self.key_link = ['base','link_3.0_tip','link_7.0_tip','link_11.0_tip','link_15.0_tip']
        self.key_link_ = ['link_15.0_tip','link_3.0_tip','link_7.0_tip','link_11.0_tip','base']

        self.collision_links = ['base_link', 'link_1.0', 'link_2.0', 'link_3.0_tip', 'link_5.0', 'link_6.0', 'link_7.0_tip', 'link_9.0', 'link_10.0', 'link_11.0_tip', 'link_12.0', 'link_14.0', 'link_15.0', 'link_15.0_tip']
        self.contact_links = ['link_3.0_tip', 'link_7.0_tip', 'link_11.0_tip', 'link_15.0_tip']
        # self.self_collision_links = {'link_3.0_tip': ['link_1.0', 'link_2.0', 'link_3.0_tip'],
        #                             'link_7.0_tip': ['link_5.0', 'link_6.0', 'link_7.0_tip'],
        #                             'link_11.0_tip': ['link_9.0', 'link_10.0', 'link_11.0_tip'],
        #                             'link_15.0_tip': ['link_12.0', 'link_14.0', 'link_15.0', 'link_15.0_tip']}
        self.self_collision_links = {'link_3.0_tip': ['link_3.0_tip'],
                                    'link_7.0_tip': ['link_7.0_tip'],
                                    'link_11.0_tip': ['link_11.0_tip']}
                                    # 'link_15.0_tip': ['link_15.0_tip']}
        self.dense_sample_link = ['link_3.0_tip', 'link_7.0_tip', 'link_11.0_tip', 'link_15.0_tip']
        self.contact_points_file = json.load(open(osp.join(osp.dirname(__file__), '..', 'model', 'urdf', 'contact_allegro.json')))
        self.calibration_rotation = euler_to_mat(torch.tensor([-torch.pi/2,torch.pi,torch.pi/2]), False).view(1,3,3).to(self.device).float()
        self.calibration_translation = torch.tensor([0.0500, 0.0000, 0.1000]).view(1,3,1).to(self.device)
        self.device = device
        self.chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(dtype=torch.float32, device=self.device)
        
        self.minuend =    torch.tensor([1,2,3,4, 1,2,3, 2,3, 2], dtype = torch.long, device=self.device)
        self.subtrahend = torch.tensor([0,0,0,0, 4,4,4, 1,1, 3], dtype = torch.long, device=self.device)      
        self.joint_lower = torch.tensor([[-0.47, -0.196, -0.174, -0.227, -0.47, -0.196, -0.174, -0.227, -0.47, -0.196, -0.174, -0.227
                                        , 0.263, -0.105, -0.189, -0.162]] ,dtype=torch.float32, device=self.device).repeat([self.batch, 1])
        self.joint_upper = torch.tensor([[0.47, 1.61, 1.709, 1.618, 0.47, 1.61, 1.709, 1.618, 0.47, 1.61, 1.709, 1.618,
                                         1.396, 1.163, 1.644, 1.719]] ,dtype=torch.float32, device=self.device).repeat([self.batch, 1])  
        self.q_init = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],dtype=torch.float32, device=self.device).repeat([self.batch, 1])       
        for frame_name in self.chain.get_frame_names(exclude_fixed=False):
            frame = self.chain.find_frame(frame_name)
            for link_vis in frame.link.visuals:
                if link_vis.geom_type == "mesh":
                    filename = link_vis.geom_param[0].split("/")[-1]
                    mesh = tm.load(osp.join(mesh_path, filename), force='mesh', process=False)
                elif link_vis.geom_type == None:
                    continue
                else:
                    print(link_vis)
                    raise NotImplementedError
                try:
                    scale = np.array(link_vis.geom_param[1]).reshape([1,3])
                except:
                    scale = np.array([[1,1,1]])
                try:
                    offset = link_vis.offset.get_matrix()
                    pos = offset[:,:3,3].numpy()
                    rot = offset[:,:3,:3].squeeze().numpy()
                except:
                    offset = pk.Transform3d().get_matrix()
                    pos = offset[:,:3,3].numpy()
                    rot = offset[:,:3,:3].squeeze().numpy()
                print(frame.link.name, filename)
                self.meshV[frame.link.name] = np.array(mesh.vertices) * scale
                self.meshV[frame.link.name] = np.matmul(rot, self.meshV[frame.link.name].T).T + pos
                self.meshF[frame.link.name] = np.array(mesh.faces)      
                if frame.link.name == 'base_link':
                    pts, pts_face_index = tm.sample.sample_surface(mesh=mesh, count=512)
                elif frame.link.name in self.dense_sample_link:
                    pts, pts_face_index = tm.sample.sample_surface(mesh=mesh, count=128)
                else:
                    pts, pts_face_index = tm.sample.sample_surface(mesh=mesh, count=128)
                pts_normal = np.array([mesh.face_normals[i] for i in pts_face_index], dtype=float)
                pts *= scale
                pts = np.matmul(rot, pts.T).T + pos
                pts = np.concatenate([pts, np.ones([len(pts), 1])], axis=-1)
                pts_normal = np.concatenate([pts_normal, np.ones([len(pts_normal), 1])], axis=-1)
                self.surface_points[frame.link.name] = torch.from_numpy(pts).to(
                    self.device).float().unsqueeze(0).repeat(self.batch, 1, 1)
                self.surface_points_normal[frame.link.name] = torch.from_numpy(pts_normal).to(
                    self.device).float().unsqueeze(0).repeat(self.batch, 1, 1)
                
                #contact points
                if frame.link.name in self.contact_links:
                    contact_pack = np.array(self.contact_points_file[frame.link.name])
                    cont_pts = contact_pack[0]*scale
                    cont_pts = np.matmul(rot, cont_pts.T).T + pos
                    cont_pts = torch.cat([torch.from_numpy(cont_pts).to(self.device).float(), torch.ones(cont_pts.shape[0],1).to(self.device).float()], dim=-1)
                    self.robot_contact_points[frame.link.name] = cont_pts.unsqueeze(0).repeat(self.batch, 1, 1)
                    cont_nls = contact_pack[1]
                    cont_nls = np.matmul(rot, cont_nls.T).T
                    self.robot_contact_normal[frame.link.name] = torch.from_numpy(cont_nls).to(self.device).float().unsqueeze(0).repeat(self.batch, 1, 1)

                             
class Barrett(CtcRobot):
    def __init__(self, batch, device):
        super(Barrett, self).__init__(batch, device)
        urdf_path = osp.join(osp.dirname(__file__), '..', 'model', 'urdf', 'barrett_adagrasp','model.urdf')
        mesh_path = osp.join(osp.dirname(__file__), '..', 'model', 'urdf', 'barrett_adagrasp','meshes')
        self.mesh_path = mesh_path

        self.robot_model = 'Barrett'
        self.contacts_num = 3
        self.joint_num = 8
        self.minimal_dist = 0.005

        self.key_link = ['bh_base_link','f13_tip','f23_tip','f33_tip']
        self.key_link_ = ['f33_tip','f13_tip','f23_tip','bh_base_link']
        self.collision_links = ['bh_base_link', 'bh_finger_32_link', 'bh_finger_33_link', 'bh_finger_11_link', 'bh_finger_12_link', 'bh_finger_13_link', 'bh_finger_21_link', 'bh_finger_22_link', 'bh_finger_23_link']
        self.contact_links = ['bh_finger_13_link','bh_finger_23_link','bh_finger_33_link']
        # self.self_collision_links = {'bh_finger_13_link': ['bh_finger_11_link', 'bh_finger_12_link', 'bh_finger_13_link'],
        #                             'bh_finger_23_link': ['bh_finger_21_link', 'bh_finger_22_link', 'bh_finger_23_link'],
        #                             'bh_finger_33_link': ['bh_finger_31_link', 'bh_finger_32_link', 'bh_finger_33_link']}
        self.self_collision_links = {'bh_finger_13_link': ['bh_finger_13_link'],
                                    'bh_finger_23_link': ['bh_finger_23_link'],
                                    'bh_finger_33_link': ['bh_finger_33_link']}
        self.dense_sample_link = ['bh_finger_13_link','bh_finger_23_link','bh_finger_33_link']
        self.contact_points_file = json.load(open(osp.join(osp.dirname(__file__), '..', 'model', 'urdf', 'contact_barrett.json')))
        self.calibration_rotation = euler_to_mat(torch.tensor([0,torch.pi/2,torch.pi]), False).view(1,3,3).to(self.device).float()
        self.calibration_translation = torch.tensor([0.0000, 0.0000, 0.0000]).view(1,3,1).to(self.device) # 0, 0, 0.1
        self.device = device
        self.chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(dtype=torch.float32, device=self.device)
        
        self.minuend =    torch.tensor([1,2,3, 1,2, 1], dtype = torch.long, device=self.device)
        self.subtrahend = torch.tensor([0,0,0, 3,3, 2], dtype = torch.long, device=self.device)      
        self.joint_lower = torch.tensor([[0,0,0,0,0,0,0,0]]
                                         ,dtype=torch.float32, device=self.device).repeat([self.batch, 1])
        self.joint_upper = torch.tensor([[2.44,0.84,3.141,2.44,0.84,3.141,2.44,0.84]],dtype=torch.float32, device=self.device).repeat([self.batch, 1])
        self.q_init = torch.tensor([[0,0,0,0,0,0,0,0]]
                                         ,dtype=torch.float32, device=self.device).repeat([self.batch, 1])
        for frame_name in self.chain.get_frame_names(exclude_fixed=False):
            frame = self.chain.find_frame(frame_name)
            for link_vis in frame.link.visuals:
                if link_vis.geom_type == "mesh":
                    filename = link_vis.geom_param[0].split("/")[-1]
                    mesh = tm.load(osp.join(mesh_path, filename), force='mesh', process=False)
                elif link_vis.geom_type == None:
                    continue
                else:
                    print(link_vis)
                    raise NotImplementedError
                try:
                    scale = np.array(link_vis.geom_param[1]).reshape([1,3])
                except:
                    scale = np.array([[1,1,1]])
                try:
                    offset = link_vis.offset.get_matrix()
                    pos = offset[:,:3,3].numpy()
                    rot = offset[:,:3,:3].squeeze().numpy()
                except:
                    offset = pk.Transform3d().get_matrix()
                    pos = offset[:,:3,3].numpy()
                    rot = offset[:,:3,:3].squeeze().numpy()
                # print(frame.link.name, filename)
                self.meshV[frame.link.name] = np.array(mesh.vertices) * scale
                self.meshV[frame.link.name] = np.matmul(rot, self.meshV[frame.link.name].T).T + pos
                self.meshF[frame.link.name] = np.array(mesh.faces) 
                if frame.link.name == 'bh_base_link':
                    pts, pts_face_index = tm.sample.sample_surface(mesh=mesh, count=512)
                elif frame.link.name in self.dense_sample_link:
                    pts, pts_face_index = tm.sample.sample_surface(mesh=mesh, count=32)
                else:      
                    pts, pts_face_index = tm.sample.sample_surface(mesh=mesh, count=32)
                pts_normal = np.array([mesh.face_normals[i] for i in pts_face_index], dtype=float)
                pts *= scale
                pts = np.matmul(rot, pts.T).T + pos

                pts = np.concatenate([pts, np.ones([len(pts), 1])], axis=-1)
                pts_normal = np.concatenate([pts_normal, np.ones([len(pts_normal), 1])], axis=-1)
                self.surface_points[frame.link.name] = torch.from_numpy(pts).to(
                    self.device).float().unsqueeze(0).repeat(self.batch, 1, 1)
                self.surface_points_normal[frame.link.name] = torch.from_numpy(pts_normal).to(
                    self.device).float().unsqueeze(0).repeat(self.batch, 1, 1)
                
                #contact
                if frame.link.name in self.contact_links:
                    contact_pack = np.array(self.contact_points_file[frame.link.name])
                    cont_pts = contact_pack[0]*scale
                    cont_pts = np.matmul(rot, cont_pts.T).T + pos
                    cont_pts = torch.cat([torch.from_numpy(cont_pts).to(self.device).float(), torch.ones(cont_pts.shape[0],1).to(self.device).float()], dim=-1)
                    self.robot_contact_points[frame.link.name] = cont_pts.unsqueeze(0).repeat(self.batch, 1, 1)
                    cont_nls = contact_pack[1]
                    cont_nls = np.matmul(rot, cont_nls.T).T
                    self.robot_contact_normal[frame.link.name] = torch.from_numpy(cont_nls).to(self.device).float().unsqueeze(0).repeat(self.batch, 1, 1)


class Robotiq(CtcRobot):
    def __init__(self, batch, device):
        super(Robotiq, self).__init__(batch, device)
        urdf_path = osp.join(osp.dirname(__file__), '..', 'model', 'urdf', 'robotiq_arg85','urdf','robotiq_arg85_description.urdf')
        mesh_path = osp.join(osp.dirname(__file__), '..', 'model', 'urdf', 'robotiq_arg85','meshes')
        self.mesh_path = mesh_path

        # urdf_path = osp.join(osp.dirname(__file__), '..', 'model', 'urdf', 'Robotiq_85', 'robotiq_85.urdf')
        # mesh_path = osp.join(osp.dirname(__file__), '..',  'model', 'urdf', 'Robotiq_85', 'robotiq_85', 'visual')
        # urdf_path = osp.join(osp.dirname(__file__), '..', 'model', 'urdf', 'robotiq_85v2', 'robotiq_2f_85_v3.urdf')
        # mesh_path = osp.join(osp.dirname(__file__), '..',  'model', 'urdf', 'robotiq_85v2', 'robotiq_2f_85')

        self.robot_model = 'Robotiq'
        self.contacts_num = 2
        #self.joint_num = 6
        self.joint_num = 1
        self.minimal_dist = 0.005
        self.key_link = ['robotiq_85_base_link','left_inner_finger_tip','right_inner_finger_tip']
        self.key_link_ = ['left_inner_finger_tip','right_inner_finger_tip','robotiq_85_base_link']
        self.collision_links=['robotiq_85_base_link','left_inner_finger','right_inner_finger','left_inner_knuckle','right_inner_knuckle']
        self.contact_links = ['left_inner_finger','right_inner_finger']
        self.self_collision_links = {'left_inner_finger': ['left_inner_finger','left_inner_knuckle'],
                                        'right_inner_finger': ['right_inner_finger','right_inner_knuckle']}
        self.dense_sample_link = ['left_inner_finger','right_inner_finger']
        self.contact_points_file = json.load(open(osp.join(osp.dirname(__file__), '..', 'model', 'urdf', 'contact_robotiq_arg85.json')))
        self.calibration_rotation = euler_to_mat(torch.tensor([torch.pi/2,torch.pi,3*torch.pi/2]), False).view(1,3,3).to(self.device).float()
        self.calibration_translation = torch.tensor([0.0000, 0.0000, 0.0000]).view(1,3,1).to(self.device)
        self.device = device
        self.chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(dtype=torch.float32, device=self.device)
        # self.joint_upper = torch.tensor([[0.,     0.,  0.72, 0.,  0.72,0.]]
        #                                  ,dtype=torch.float32, device=self.device).repeat([self.batch, 1])
        # self.joint_lower = torch.tensor([[-0.72, -0.72,0.0, -0.72,0.0,   -0.72]],dtype=torch.float32, device=self.device).repeat([self.batch, 1])
        # self.joint_lower = torch.tensor([[-0.8, -0.8757, 0.0, -0.81, -0.0, -0.8757]]
        #                                  ,dtype=torch.float32, device=self.device).repeat([self.batch, 1])
        # self.joint_upper = torch.tensor([[0.0,   0.0,    0.8757, 0.0, 0.8757, 0.0]],dtype=torch.float32, device=self.device).repeat([self.batch, 1])
        self.joint_lower = torch.tensor([[0.0]],dtype=torch.float32, device=self.device).repeat([self.batch, 1])
        self.joint_upper = torch.tensor([[0.725]],dtype=torch.float32, device=self.device).repeat([self.batch, 1])
        self.expend_ = torch.tensor([[-1, -1, 1, -1, 1, -1]]).to(self.device).float().repeat([self.batch, 1])
        self.q_init = torch.tensor([[0.0]],dtype=torch.float32, device=self.device).repeat([self.batch, 1])
        #self.q_init = self.q * self.expend_
        self.minuend =    torch.tensor([1,2, 1], dtype = torch.long, device=self.device)
        self.subtrahend = torch.tensor([0,0, 2], dtype = torch.long, device=self.device)    
        for frame_name in self.chain.get_frame_names(exclude_fixed=False):
            frame = self.chain.find_frame(frame_name)
            for link_vis in frame.link.visuals:
                if link_vis.geom_type == "mesh":
                    filename = link_vis.geom_param[0].split("/")[-1]
                    mesh = tm.load(osp.join(mesh_path, filename), force='mesh', process=False)
                elif link_vis.geom_type == "box":
                    mesh = tm.primitives.Box(extents=link_vis.geom_param)
                elif link_vis.geom_type == None:
                    continue
                else:
                    print(link_vis)
                    raise NotImplementedError
                try:
                    scale = np.array(link_vis.geom_param[1]).reshape([1,3])
                except:
                    #print(frame.link.name)
                    scale = np.array([[1,1,1]])
                try:
                    offset = link_vis.offset.get_matrix()
                    pos = offset[:,:3,3].numpy()
                    rot = offset[:,:3,:3].squeeze().numpy()
                except:
                    offset = pk.Transform3d().get_matrix()
                    pos = offset[:,:3,3].numpy()
                    rot = offset[:,:3,:3].squeeze().numpy()
                # print(frame.link.name, filename)

                self.meshV[frame.link.name] = np.array(mesh.vertices) * scale
                self.meshV[frame.link.name] = np.matmul(rot, self.meshV[frame.link.name].T).T + pos
                self.meshF[frame.link.name] = np.array(mesh.faces)        
                if frame.link.name == 'robotiq_85_base_link':
                    pts, pts_face_index = tm.sample.sample_surface(mesh=mesh, count=512)
                elif frame.link.name in self.dense_sample_link:
                    pts, pts_face_index = tm.sample.sample_surface(mesh=mesh, count=32)
                else:
                    pts, pts_face_index = tm.sample.sample_surface(mesh=mesh, count=32)
                pts_normal = np.array([mesh.face_normals[i] for i in pts_face_index], dtype=float)
                pts *= scale
                pts = np.matmul(rot, pts.T).T + pos
                pts = np.concatenate([pts, np.ones([len(pts), 1])], axis=-1)
                pts_normal = np.concatenate([pts_normal, np.ones([len(pts_normal), 1])], axis=-1)
                self.surface_points[frame.link.name] = torch.from_numpy(pts).to(
                    self.device).float().unsqueeze(0).repeat(self.batch, 1, 1)
                self.surface_points_normal[frame.link.name] = torch.from_numpy(pts_normal).to(
                    self.device).float().unsqueeze(0).repeat(self.batch, 1, 1)
                #contact
                if frame.link.name in self.contact_links:
                    contact_pack = np.array(self.contact_points_file[frame.link.name])
                    cont_pts = contact_pack[0]*scale
                    cont_pts = np.matmul(rot, cont_pts.T).T + pos
                    cont_pts = torch.cat([torch.from_numpy(cont_pts).to(self.device).float(), torch.ones(cont_pts.shape[0],1).to(self.device).float()], dim=-1)
                    self.robot_contact_points[frame.link.name] = cont_pts.unsqueeze(0).repeat(self.batch, 1, 1)
                    cont_nls = contact_pack[1]
                    cont_nls = np.matmul(rot, cont_nls.T).T
                    self.robot_contact_normal[frame.link.name] = torch.from_numpy(cont_nls).to(self.device).float().unsqueeze(0).repeat(self.batch, 1, 1)


class human():
    def __init__(self, device = 'cuda'):
        self.contact_zones= {0: [350, 355, 329, 332, 349, 354, 343, 327, 351, 353, 348, 328, 347, 326, 337, 346], #index
                             1: [462, 439, 467, 442, 461, 466, 455, 437, 459, 438, 465, 460, 436, 463, 449, 434], 
                             2: [573, 550, 578, 577, 553, 572, 566, 570, 576, 549, 548, 547, 574, 571, 546, 569], 
                             3: [690, 689, 667, 695, 670, 694, 664, 683, 688, 665, 693, 691, 666, 687, 661, 663], 
                             4: [743, 738, 768, 740, 763, 737, 766, 767, 764, 734, 735, 762, 745, 761, 717, 765], #thumb
                             }
        # self.contact_zones= {0: [338,352,319,353,324,327,355,354,348,349,350,332,347,343,328,345,344,342,301,330], 
        #                      1: [463,444,428,434,467,466,460,442,433,439,462,458,459,455,438,435,432,436,456,453], 
        #                      2: [575,574,571,577,572,570,555,576,578,573,566,548,553,550,549,545,547,546,552,551], 
        #                      3: [692,691,677,688,686,675,677,676,642,687,672,693,695,690,683,662,665,664,667,666], 
        #                      4: [751,764,767,746,760,745,738,743,740,739,735,734,737,736,742,741,753,755,714,711], 
        #                      }
        """
        0: palm 0: [98, 97, 774, 96, 244, 607, 604, 772, 595, 776, 596, 775, 770, 769, 243, 99, 771, 777, 73], 
        1: index 1: [355, 330, 348, 351, 343, 317, 320, 324, 325, 350, 332, 333, 354, 318, 337, 353, 349, 347, 328, 327, 323, 319, 329, 352, 326, 338, 342, 339],
        2: [465, 467, 429, 463, 437, 440, 436, 433, 444, 443, 461, 442, 439, 462, 435, 434, 438, 455, 466], 
        3: [553, 548, 547, 550, 578, 573, 566], 
        4: [690, 688, 677, 672, 689, 678, 684, 691, 686, 683, 692, 667, 666, 687, 664, 693, 665, 662, 671, 694, 657, 695, 670, 682, 661], 
        5: [743, 741, 760, 763, 739, 767, 762, 753, 754, 761, 756, 766, 737, 764, 768, 755, 740]}
        """
        self.device = device
        self.hand_mesh = None

    def load_hand_mesh(self, mesh_v, mesh_f, joints):
        if not isinstance(mesh_v, torch.Tensor):
            verts = torch.tensor(mesh_v, dtype=torch.float32)
        else:
            verts = mesh_v
        if not isinstance(mesh_f, torch.Tensor):
            faces = torch.tensor(mesh_f, dtype=torch.long)
        else:
            faces = mesh_f.to(self.device)
        self.hand_mesh = Meshes(verts=verts, faces=faces)
        self.joints = joints
    
    def get_joints(self):
        return self.joints
    
    def get_fingertips(self):
        tip_index =torch.tensor([4,8,12,16,20], dtype=torch.long, device=self.device)
        return self.joints[:,tip_index,:]
        # return self.joints[tip_index]

    def get_keypoints(self):
        key_indices = torch.tensor([0, 8, 12, 16, 20, 4], dtype=torch.long, device=self.device)
        return self.joints[:,key_indices,:]
    
    def get_contact_zones(self):
        """Assume we only have one hand mesh for now"""
        assert self.hand_mesh is not None, f"Please load the hand mesh first"
        vertices = self.hand_mesh.verts_list()[0]
        cnt_points = []
        for zone in self.contact_zones:
            cnt_points.append(vertices[self.contact_zones[zone]])
        cnt_p = torch.stack(cnt_points).to(self.device)
        return cnt_p

    def get_contact_normals(self):
        """Assume we only have one hand mesh for now 1,N,3"""
        assert self.hand_mesh is not None, f"Please load the hand mesh first"
        normals = self.hand_mesh.verts_normals_list()[0]
        cnt_normals = []
        for zone in self.contact_zones:
            cnt_normals.append(normals[self.contact_zones[zone]])
        cnt_n = torch.stack(cnt_normals).to(self.device)
        return cnt_n
    
    def get_contact_points_normals_packed(self):
        cnt_p = self.get_contact_zones()
        cnt_n = self.get_contact_normals()
        return torch.cat([cnt_p, cnt_n], dim=2).reshape(-1, 6)

    def get_contact_points(self, zone = None):

        contact_points = self.hand_mesh.vertices[self.contact_zones[zone]]
        return contact_points

    def get_go_mesh(self,color='white',opacity = 1):
        vertices = self.hand_mesh.verts_list()[0].cpu().numpy()
        faces = self.hand_mesh.faces_list()[0].cpu().numpy()

        mesh = go.Mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2], 
                        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], 
                        color=color, opacity=opacity)
        edges = set()
        for face in faces:
            for e1, e2 in [(0, 1), (1, 2), (2, 0)]:
                edge = tuple(sorted([face[e1], face[e2]]))  # Ensure consistent ordering
                edges.add(edge)

        # Convert edges into plot data
        edge_x, edge_y, edge_z = [], [], []
        for e1, e2 in edges:
            x0, y0, z0 = vertices[e1]
            x1, y1, z1 = vertices[e2]
            edge_x += [x0, x1, None]  # None breaks the line
            edge_y += [y0, y1, None]
            edge_z += [z0, z1, None]

        # Create wireframe using Scatter3d
        wireframe = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode="lines",
            line=dict(color='black', width=1.5),
            name="Wireframe",
            opacity=0.7
        )        
        return [mesh, wireframe]
