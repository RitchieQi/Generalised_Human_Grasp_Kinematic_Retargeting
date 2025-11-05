from dataset_opt import dexycb_testfullfeed
from optimisation.robot import CtcRobot, Shadow, Allegro, Barrett, Robotiq
import torch
from tqdm import tqdm
from optimisation.objects import object_sdf
from plotly import graph_objects as go
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
from typing import List
import time
import trimesh as tm
from scipy.optimize import linear_sum_assignment
import optimisation.icp as icp
import itertools
import os
import json
osp = os.path
import traceback
import numpy as np
import torch.multiprocessing as mp


class SubsetRestartSampler(Sampler):
    def __init__(self, start, end):
        self.indices = range(start, end)
    
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)
    
class TensorIndexSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices.long()

    def __iter__(self):
        return iter(self.indices.tolist())

    def __len__(self):
        return len(self.indices)

class EarlyStopping:
    def __init__(self, patience=None, min_delta=None):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = None
        self.early_stop = False
        print("patience", self.patience)
    
    def reset(self):
        self.counter = 0
        self.min_loss = None
        self.early_stop = False
    
    def ealy_stop(self, loss):
        if self.min_loss is None:
            self.min_loss = loss
        elif loss < self.min_loss - self.min_delta:
            self.min_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        


class Optimization:
    """
    Currently we still have problems with the parallelization of the optimization process, 
    due to the out-of-sequence test dataset and variance in the posture of the objects.
    """
    def __init__(self, 
                 robot: CtcRobot,
                 dataset: Dataset,
                 device: str,
                 maximum_iter: list,
                 visualize: bool = False,
                 repeat: int = 1,
                 mu: float = 0.9,
                 task: str = "GH",
                 source: str = "ycb",
                 ):
        self.robot = robot
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        self.device = device
        self.maximum_iter = maximum_iter
        self.visualize = visualize
        self.obj_model = object_sdf(device=self.device, robot_contact=self.robot.contacts_num, contact_threshold=0.020)
        self.contact_normal = None
        self.actual_contact = None
        self.repeat = repeat
        self.mu = mu
        
        if task == "GH" and source == "ycb":
            self.result_dir = osp.join(osp.dirname(__file__), "results", self.robot.robot_model, "mu%.1f" % self.mu)
        elif task == "NV" and source == "ycb":
            self.result_dir = osp.join(osp.dirname(__file__), "results_nv", self.robot.robot_model, "mu%.1f" % self.mu)
        elif task == "GH" and source == "sdf":
            self.result_dir = osp.join(osp.dirname(__file__), "results_sdf", self.robot.robot_model, "mu%.1f" % self.mu)
        elif task == "NV" and source == "sdf":
            self.result_dir = osp.join(osp.dirname(__file__), "results_nv_sdf", self.robot.robot_model, "mu%.1f" % self.mu)
        else:
            raise ValueError("task should be either GH or NV")
        self.task = task
        self.source = source
        if not osp.exists(self.result_dir):
            print("create result directory")
            os.makedirs(self.result_dir)
            
    def lr_schedule_fc(self, optimizer, epoch):
        inital_lr = 0.005
        interval = 1000
        factor = 0.6
        lr = inital_lr * (factor ** (epoch // interval))
        for i, param in enumerate(optimizer.param_groups):
            param['lr'] = lr

    def lr_schedule_gq(self, optimizer, epoch):
        inital_lr = 0.005
        interval = 1000
        factor = 0.6
        lr = inital_lr * (factor ** (epoch // interval))
        for i, param in enumerate(optimizer.param_groups):
            param['lr'] = lr
    
    def lr_schedule_q(self, optimizer, epoch):
        inital_lr = 0.1
        interval = 50
        factor = 0.5
        lr = inital_lr * (factor ** (epoch // interval))
        for i, param in enumerate(optimizer.param_groups):
            param['lr'] = lr

    def lr_schedule_q_2(self, optimizer, epoch):
        inital_lr = 0.1
        interval = 100
        factor = 0.5
        lr = inital_lr * (factor ** (epoch // interval))
        for i, param in enumerate(optimizer.param_groups):
            param['lr'] = lr

    def write_json(self, data, filename):
        def convert_tensor_to_list(data):
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    data[key] = value.tolist()
            return data
        file_name = osp.join(self.result_dir, filename+".json")
        print(file_name)
        with open(file_name, 'w') as f:
            json.dump(convert_tensor_to_list(data), f, indent=4)

    def initialization(self, 
                       obj_mesh_v: torch.Tensor, 
                       obj_mesh_f: torch.Tensor,
                       obj_centre: torch.Tensor,
                       hand_mesh_v: torch.Tensor, 
                       hand_mesh_f: torch.Tensor,
                       joints: torch.Tensor,
                       transformation: torch.Tensor,
                       ): #add object information
        #reset origin to the object centre
        obj_mesh_v = obj_mesh_v - obj_centre
        hand_mesh_v = hand_mesh_v - obj_centre
        joints = joints - obj_centre
        transformation[:, :3, 3] = transformation[:, :3, 3] - obj_centre
        
        self.obj_model.loss.mu = torch.tensor(self.mu).float().to(self.device)
        print("mu", self.obj_model.loss.mu.item())
        self.robot.init_q(transformation)
        self.q = self.robot.q
        self.robot.forward_kinematics(self.q)
        self.opt_q = torch.optim.Adam([self.q], lr=0.1)
        self.obj_model.reset([obj_mesh_v, obj_mesh_f], [hand_mesh_v, hand_mesh_f], joints)
        #self.obj_model.hand_contact_cluster_mano_agnostic()
        #self.obj_model.hand_contact_cluster_v2()
        self.obj_model.hand_contact_cluster_hdbscan()
        # x_ = torch.randn((1,self.obj_model.contact_target.size(0),3),dtype=torch.float32,requires_grad=True, device=self.device)*0.05
        x_ = self.obj_model.contact_target.unsqueeze(0)
        # print("target", x_.size())
        self.x = x_.clone().detach().requires_grad_(True)
        print("self x", self.x)
        w = torch.ones((1,self.obj_model.contact_target.size(0),4),dtype=torch.float32, device=self.device)*0.25
        self.w = w.clone().detach().requires_grad_(True)
        self.robot.row_ind = None
        self.robot.col_ind = None
        self.opt_fc = torch.optim.Adam([self.x, self.w], lr=0.001)
        self.opt_gq = torch.optim.Adam([self.w], lr=0.001)
        # self.opt_gq = torch.optim.Adam([self.x, self.w], lr=0.001)

        self.opt_gf = torch.optim.Adam([self.w], lr=0.005)
        self.opt_gqre = torch.optim.Adam([self.w], lr=0.001)
        
        self.fc_early_stop = EarlyStopping(patience=100, min_delta=0)
        self.gq_early_stop = EarlyStopping(patience=2000, min_delta=0)
        if self.robot.robot_model == "Shadow" and self.robot.robot_model == "Allegro":
            self.q_early_stop = EarlyStopping(patience=20, min_delta=0.01)
        self.q_early_stop = EarlyStopping(patience=100, min_delta=0.001)

        if self.task == "NV":
            self.target = torch.cat((self.obj_model.hand_model.get_keypoints().squeeze(), torch.tensor([0., 0., 0.], device=self.device).view(1, 3)), dim=0)

        print('x',self.x)
    
    def penetration_check(self, q):
        obj_points = self.obj_model.obj_points
        obj_normals = self.obj_model.obj_normals
        hand_points_ = self.robot.get_surface_points_updated(q)
        npts_obj = obj_points.shape[0]
        npts_hand = hand_points_.shape[0]
        #print("dim check", obj_points.shape, hand_points_.shape)
        obj_points = obj_points.unsqueeze(0).repeat(npts_hand, 1, 1)
        hand_points = hand_points_.unsqueeze(1).repeat(1, npts_obj, 1)

        dist = (hand_points - obj_points).norm(dim=2) #npts_hand, npts_obj, 1
        hand_obj_dist, hand_obj_idx = dist.min(dim=1) #npts_hand, 1
        hand_obj_point = torch.stack([self.obj_model.obj_points[x, :] for x in hand_obj_idx], dim=0) #npts_hand, 3
        hand_obj_normal = torch.stack([obj_normals[x, :] for x in hand_obj_idx], dim=0)

        hand_object_signs = ((hand_obj_point - hand_points_) * hand_obj_normal).sum(dim=1)
        hand_object_signs = (hand_object_signs > 0).float()
        penetration = (hand_object_signs * hand_obj_dist).mean()
        return penetration

    def self_penetration_check(self, q):
        def single_penetration(pointsA, pointwnB):
            """
            input:  pointsA: n x 3
                    pointwnB: (n x 3, n x 3) 
            """
            pointsB, normalsB = pointwnB
            nptsA = pointsA.size(0)
            nptsB = pointsB.size(0)
            
            pointsB_expand = pointsB.unsqueeze(0).repeat(nptsA, 1, 1)
            pointsA_expand = pointsA.unsqueeze(1).repeat(1, nptsB, 1)
            dist = (pointsA_expand - pointsB_expand).norm(dim=2)
            dist, idx = dist.min(dim=1)
            pointsA_B = torch.stack([pointsB[x, :] for x in idx], dim=0)
            normalsA_B = torch.stack([normalsB[x, :] for x in idx], dim=0)
            
            signs = ((pointsA_B - pointsA) * normalsA_B).sum(dim=1)
            signs = (signs > 0).float()
            penetration = (signs * dist).mean()
            return penetration
            
        finger_points, finger_normals = self.robot.get_finger_surface_points_normal_updated(q)
        finger_num = len(finger_points)
        idx_list = list(itertools.combinations(range(finger_num), 2))
        penetration = 0.00
        for indices in idx_list:
            penetration_finger = single_penetration(finger_points[indices[0]], (finger_points[indices[1]], finger_normals[indices[1]]))
            penetration += penetration_finger
        return penetration
    
    def robot_object_contact_check(self, q):
        tip_points = self.robot.get_fingertip_surface_points_updated(q)
        #idx_list = [*range(0, self.x.size(1))]
        #idx_list.append(-1)
        #idx_list = torch.tensor(idx_list, dtype=torch.long, device=self.device)
        #print("idx_list", idx_list)
        utilized_tip = tip_points[: self.x.size(1)-1]
        utilized_tip.append(tip_points[-1])
        robot_contact = []
        obj_contact = []
        for fingertip in utilized_tip:
            fingertip = fingertip.squeeze()
            val, dire = self.obj_model.pv_sdf(fingertip)
            val_min, idx = torch.min(val, dim=0)
            #print("val_min", val_min)
            # if val_min <= 0:
            #     #val_ = val[(val < 0.001) & (val > -0.001)]
            #     indices = torch.where((val < 0.001) & (val > -0.001))
            #     robot_point = fingertip[indices].mean(dim=0).unsqueeze(0)
            # else:
            #     robot_point = fingertip[idx]
            #     robot_point, _, _ = tm.proximity.closest_point(self.obj_model.obj_mesh, robot_point.unsqueeze(0).detach().cpu().numpy())
            robot_point = fingertip[idx]
            direction = dire[idx]
            obj_point = robot_point - val_min * direction
            #lay = tm.proximity.longest_ray(self.obj_model.obj_mesh, robot_point.unsqueeze(0).detach().cpu().numpy(), direction.unsqueeze(0).detach().cpu().numpy())
            # obj_point, _, _ = tm.proximity.closest_point(self.obj_model.obj_mesh, robot_point.unsqueeze(0).detach().cpu().numpy())
            
            # robot_contact.append(torch.tensor(robot_point, dtype=torch.float32).to(self.device))
            # obj_contact.append(torch.tensor(obj_point, dtype=torch.float32).to(self.device))
            robot_contact.append(robot_point.unsqueeze(0))
            obj_contact.append(obj_point.unsqueeze(0))
        robot_contact = torch.stack(robot_contact, dim=1)
        print("robot_contact", robot_contact.size())
        obj_contact = torch.stack(obj_contact, dim=1)
        return robot_contact, obj_contact
    
    def step(self, loss_func, data, optimizer):
        optimizer.zero_grad()
        loss = loss_func(*data)["loss"]
        loss.backward()
        optimizer.step()
        return loss_func(*data)
    
    def step_with_verbose(self, loss_func, data, optimizer, pbar):
        optimizer.zero_grad()
        loss = loss_func(*data)["loss"]
        loss.backward()
        optimizer.step()
        description = [f"{key}: {value}" for key, value in loss_func(data).items()]
        pbar.set_description(", ".join(description))
        return loss_func(*data)
    
    def fc_optimization(self):
        converge = False
        pbar = tqdm(enumerate(range(self.maximum_iter[0])), total=self.maximum_iter[0])
        #for i, _ in enumerate(range(self.maximum_iter[0])):
        for i, _ in pbar:
            if converge:
                break
            data = self.obj_model.loss_fc(self.x, self.w)
            self.opt_fc.zero_grad()
            loss_fc = data["loss"]
            loss_fc.backward()
            self.opt_fc.step()
            if torch.allclose(data["sdf"],torch.tensor(0, dtype=torch.float), atol=1e-6) and torch.allclose(data["Gf"],torch.tensor(0, dtype=torch.float), atol=1e-3) and torch.allclose(data["distance"],torch.tensor(0, dtype=torch.float), atol=1e-3):
                 converge = True

    def gq_optimization(self):
        pbar = tqdm(enumerate(range(self.maximum_iter[1])), total=self.maximum_iter[1])
        #for i, _ in enumerate(range(self.maximum_iter[1])):
        for i, _ in pbar:
            self.opt_gq.zero_grad()
            loss_gq = self.obj_model.grasp_quality(self.x, self.w)["loss"]
            loss_gq.backward()
            self.opt_gq.step()

    def q_optimization(self):
        pbar = tqdm(enumerate(range(self.maximum_iter[2])), total=self.maximum_iter[2])
        converge = False
        for i, _ in pbar:
        #for i, _ in enumerate(range(self.maximum_iter[2])):
            if converge:
                break
            data = self.robot.kinematics_distance(self.x, self.obj_model.hand_model.joints,self.q)
            self.opt_q.zero_grad()
            loss_q = data["loss"]
            loss_q.backward()
            self.opt_q.step()
            if torch.allclose(data["distance_loss"],torch.tensor(0, dtype=torch.float), atol=1e-3):
                 converge = True

    def fc_optimization_verbose(self, pbar):
        print('x',self.x)
        converge = False
        #pbar = tqdm(enumerate(range(self.maximum_iter[0])), total=self.maximum_iter[0])
        for i, _ in enumerate(range(self.maximum_iter[0])):
            if converge or self.fc_early_stop.early_stop:
                break
            self.lr_schedule_fc(self.opt_fc, i)
            self.opt_fc.zero_grad()
            data = self.obj_model.loss_fc(self.x, self.w)
            loss_fc = data["loss"]
            loss_fc.backward(retain_graph=True)

            self.opt_fc.step()
            description = [f"{key}: {value.item():.4f}" for key, value in data.items()]
            pbar.set_description(", ".join(description))
            self.fc_early_stop.ealy_stop(loss_fc)
            #time.sleep(0.05)
            if torch.allclose(data["sdf"],torch.tensor(0, dtype=torch.float), atol=1e-4) and torch.allclose(data["Gf"],torch.tensor(0, dtype=torch.float), atol=1e-3) and torch.allclose(data["distance"],torch.tensor(0, dtype=torch.float), atol=2e-2) and torch.allclose(data["GG"],torch.tensor(0, dtype=torch.float), atol=1e-3):
                 converge = True
        self.fc_early_stop.reset()
        return data

    def gf_optimization_verbose(self, pbar):
        converge = False
        for i, _ in enumerate(range(self.maximum_iter[1])):
            if converge:
                break
            self.lr_schedule_gq(self.opt_gf, i)
            self.opt_gf.zero_grad()
            data = self.obj_model.loss_gf(self.x, self.w)
            loss_gf = data["loss"]
            loss_gf.backward(retain_graph=True)
            self.opt_gf.step()
            description = [f"{key}: {value.item():.4f}" for key, value in data.items()]
            pbar.set_description(", ".join(description))
            if torch.allclose(data["Gf"],torch.tensor(0, dtype=torch.float), atol=1e-3):
                converge = True
        # print(description,'x',self.x,'w',self.w)
        return data

    def gq_optimization_verbose(self, pbar):
        #pbar = tqdm(enumerate(range(self.maximum_iter[1])), total=self.maximum_iter[1])
        start_r = self.obj_model.wrench_hull(self.x, self.w)
        for i, _ in enumerate(range(self.maximum_iter[1])):
            if self.gq_early_stop.early_stop:
                break
            self.lr_schedule_gq(self.opt_gq, i)
            self.opt_gq.zero_grad()
            data = self.obj_model.wrench_hull(self.x, self.w, with_x=False)
            loss_gq = data["loss"]
            loss_gq.backward(retain_graph=True)
            self.opt_gq.step()
            description = [f"{key}: {value.item():.4f}" for key, value in data.items()]
            pbar.set_description(", ".join(description))
            self.gq_early_stop.ealy_stop(data["radius"])
        # print(description,'x',self.x,'w',self.w)
        self.contact_normal = self.obj_model.pv_sdf(self.x)[1]
        data["r_start"] = start_r["radius"]
        self.gq_early_stop.reset()
        return data

    def gf_reoptimization_verbose(self, pbar):
        converge = False
        for i, _ in enumerate(range(self.maximum_iter[1])):
            if converge:
                break
            self.lr_schedule_gq(self.opt_gf, i)
            self.opt_gf.zero_grad()
            data = self.obj_model.loss_gf(self.actual_contact, self.w)
            loss_gf = data["loss"]
            loss_gf.backward(retain_graph=True)
            self.opt_gf.step()
            description = [f"{key}: {value.item():.4f}" for key, value in data.items()]
            pbar.set_description(", ".join(description))
            if torch.allclose(data["Gf"],torch.tensor(0, dtype=torch.float), atol=1e-3):
                converge = True
        # print(description,'x',self.x,'w',self.w)
        return data
    
    def gq_reoptimization_verbose(self, pbar):
        start_r = self.obj_model.wrench_hull(self.actual_contact, self.w)
        for i, _ in enumerate(range(self.maximum_iter[1])):
            if self.gq_early_stop.early_stop:
                break
            #self.lr_schedule_gq(self.opt_gqre, i)
            self.opt_gqre.zero_grad()
            data = self.obj_model.wrench_hull(self.actual_contact, self.w)
            loss_gqre = data["loss"]
            loss_gqre.backward(retain_graph=True)
            self.opt_gqre.step()
            self.gq_early_stop.ealy_stop(data["radius"])
            description = [f"{key}: {value.item():.4f}" for key, value in data.items()]
            pbar.set_description(", ".join(description))
        data["r_start"] = start_r["radius"]
        self.gq_early_stop.reset()
        return data
    
    def q_optimization_verbose(self, pbar):
        #pbar = tqdm(enumerate(range(self.maximum_iter[2])), total=self.maximum_iter[2])
        converge = False
        iter = torch.tensor(0.)
        print('q',self.q)
        for i, _ in enumerate(range(self.maximum_iter[2])):
            if converge:
                break
            # if i % 100 == 0:
            #     print('q',self.q)
            self.lr_schedule_q(self.opt_q, i)
            self.opt_q.zero_grad()
            data = self.robot.kinematics_distance_all(self.x, self.obj_model.hand_model.joints, self.q, test=True)
            #collision = self.obj_model.collision_check(self.robot.robot_vertices_v2(self.q))
            collision = self.penetration_check(self.q)
            self_collision = self.self_penetration_check(self.q)
            # sdf = self.obj_model.sdf_loss(self.robot.get_keypoints(self.q))
            data["collision"] = collision
            data["self_collision"] = self_collision
            # data["sdf"] = sdf
            data["loss"] += 300*collision
            data["loss"] += 500*self_collision
            # data["loss"] += 100*sdf
            iter += 1
            data["iteration"] = iter 
            #print(self.q)
            loss_q = data["loss"]
            loss_q.backward()
            self.opt_q.step()
            description = [f"{key}: {value.item():.4f}" for key, value in data.items()]
            pbar.set_description(", ".join(description))
            #time.sleep(0.05)
            if torch.allclose(data["distance_loss"],torch.tensor(0, dtype=torch.float), atol=7e-3) and torch.allclose(collision,torch.tensor(0, dtype=torch.float), atol=1e-3):
                converge = True
        print(description,'x',self.x)

    def re_order_x(self):
        fingertips = self.obj_model.hand_model.get_fingertips()
        # indices = icp.nn_reg(self.x.squeeze().detach().cpu().numpy(), fingertips.squeeze().detach().cpu().numpy())
        dist = torch.cdist(self.x.squeeze(), fingertips.squeeze())
        row, col = linear_sum_assignment(dist.detach().cpu().numpy())
        reordered_x = self.x.squeeze().clone()
        reordered_x = reordered_x[col]
        return reordered_x.unsqueeze(0)


    def q_optimization_verbose_v2(self, pbar):
        """
        This version of the optimization function introduces the collision check in the later stage of the optimization process.
        """
        converge = False
        converge_gate = False
        dead_end = 0
        if self.robot.robot_model == "Shadow" or self.robot.robot_model == "Allegro":
            dead_end = 100
        else:
            dead_end = 200
        hand_joints=self.obj_model.hand_model.get_fingertips()
        # robot_reorder_x = self.re_order_x()
        for i, _ in enumerate(range(self.maximum_iter[2])):
            if converge or self.q_early_stop.early_stop:
                break
            if i > dead_end:
                if converge_gate:
                    break
            if self.robot.robot_model == "Shadow" or self.robot.robot_model == "Allegro":

                    self.lr_schedule_q(self.opt_q, i)
            else:

                    self.lr_schedule_q_2(self.opt_q, i)
            self.opt_q.zero_grad()
            data = self.robot.kinematics_distance_all(self.x, self.q, test=True, hand_joints=hand_joints) #[:,[4,0,1,2,3],:]
            if self.robot.robot_model != "Barrett" and self.robot.robot_model != "Robotiq":
                    self_collision = self.self_penetration_check(self.q)
                    data["self_collision"] = self_collision
                    data["loss"] += 100*self_collision
            if i > 0.001*self.maximum_iter[2]:
                collision = self.penetration_check(self.q)
                data["collision"] = collision
                data["loss"] += 500*collision
            # data["loss"] += 100*sdf
            data["iteration"] = torch.tensor(i) 
            loss_q = data["loss"]
            loss_q.backward(retain_graph=True)
            self.opt_q.step()
            description = [f"{key}: {value.item():.4f}" for key, value in data.items()]
            pbar.set_description(", ".join(description))
            self.q_early_stop.ealy_stop(data["distance_loss"])
            if torch.allclose(data["distance_loss"],torch.tensor(0, dtype=torch.float), atol=5e-3) and torch.allclose(collision,torch.tensor(0, dtype=torch.float), atol=5e-3):
                converge = True
            if torch.allclose(data["distance_loss"],torch.tensor(0, dtype=torch.float), atol=1e-2):
                converge_gate = True
                self.refine_flag = False
        self.q_early_stop.reset()
        # print(description,'x',self.x,'w',self.w)
        return data

    def q_optimization_verbose_v3(self, pbar):
        """
        This version of the optimization function introduces the collision check in the later stage of the optimization process.
        """
        converge = False
        print('q',self.q)
        for i, _ in enumerate(range(self.maximum_iter[2])):
            if converge:
                break
            self.lr_schedule_q(self.opt_q, i)
            self.opt_q.zero_grad()
            data = self.robot.kinematics_distance_wnormal((self.x,self.contact_normal), self.q)
            if self.robot.robot_model != "Barrett" and self.robot.robot_model != "Robotiq":
                self_collision = self.self_penetration_check(self.q)
                data["self_collision"] = self_collision
                data["loss"] += 1000*self_collision
            if i > 0.001*self.maximum_iter[2]:
                collision = self.penetration_check(self.q)
                data["collision"] = collision
                data["loss"] += 500*collision
            # data["loss"] += 100*sdf
            data["iteration"] = torch.tensor(i) 
            loss_q = data["loss"]
            loss_q.backward()
            self.opt_q.step()
            description = [f"{key}: {value.item():.4f}" for key, value in data.items()]
            pbar.set_description(", ".join(description))
            if torch.allclose(data["distance_loss"],torch.tensor(0, dtype=torch.float), atol=7e-3) and torch.allclose(collision,torch.tensor(0, dtype=torch.float), atol=1e-3):
                converge = True
        print(description,'x',self.x)
    
    def q_optimization_verbose_nv(self, pbar):
        """
        The implementation of the nv/uw optimization process in DexPilot (ICRA2020)
        """
        converge = False
        
        for i, _ in enumerate(range(self.maximum_iter[2])):
            if converge or self.q_early_stop.early_stop:
                break
            self.lr_schedule_q(self.opt_q, i)
            self.opt_q.zero_grad()
            
            data = self.robot.kinematics_distance_nv(self.target, self.q)
            if self.robot.robot_model != "Barrett" and self.robot.robot_model != "Robotiq":
                self_collision = self.self_penetration_check(self.q)
                data["self_collision"] = self_collision
                data["loss"] += 1000*self_collision
            
            if i > 0.001*self.maximum_iter[2]:
                collision = self.penetration_check(self.q)
                data["collision"] = collision
                data["loss"] += 500*collision
                
            data["iteration"] = torch.tensor(i)
            loss_q = data["loss"]
            loss_q.backward()
            self.opt_q.step()
            description = [f"{key}: {value.item():.4f}" for key, value in data.items()]
            pbar.set_description(", ".join(description))
            self.q_early_stop.ealy_stop(data["distance_loss"])
            if torch.allclose(data["distance_loss"],torch.tensor(0, dtype=torch.float), atol=1e-5):# and torch.allclose(collision,torch.tensor(0, dtype=torch.float), atol=1e-3):
                converge = True
        self.q_early_stop.reset()
        # print(description,'x',self.x,'w',self.w)
        return data            
               
    def draw(self, transparent_background: bool = True):
        # import icp
        import numpy as np
        data = []
        data = self.robot.get_mesh_updated(q =self.q, opacity=0.7, color='rgb(10,147,150)')
        
        # Force Closure####################################################################################
        color_points = ['rgb(35, 61, 77)','rgb(254, 127, 45)','rgb(252, 202, 70)','rgb(161, 193, 129)','rgb(97, 155, 138)','rgb(0, 95, 115)','rgb(155, 34, 38)']
        # colors_normal = ['rgb(0,95,115)','rgb(155, 34, 38)']

        we = self.obj_model.get_weighted_edges(self.x, torch.ones_like(self.w)) #b,N, 4, 3
        print("we", we.size()) 
        x = self.x.clone()
        we = we.squeeze().detach().cpu().numpy()
        x = x.squeeze().detach().cpu().numpy()
        
        for i in range(x.shape[0]):
            apex_vertex = x[i]
            base_vertices = we[i]*0.05 + x[i]
            print("apex_vertex", apex_vertex.shape, "base_vertices", base_vertices.shape)
            data.append(self.pyramids(base_vertices, apex_vertex , color=color_points[i], opacity=0.3))
            data.append(self.lines(np.tile(x[i],(4,1)), base_vertices, color=color_points[i], width=5))
            data.append(self.scatter(x[i,None], color=color_points[i], size=10))
        ##################################################################################################
        # B,N,_,_ = we.size()
        # we = we.reshape(B*N*4,3).detach().cpu().numpy()
        # x = x.repeat_interleave(4,dim=1)
        # x = x.reshape(B*N*4,3).cpu().detach().numpy()
        # print(self.w)
        # print(x.shape, we.shape)
        # for i in range(B*N*4):
        #     data.append(self.lines(x[i,None], we[i,None]+x[i,None], color='red'))
        # data.append(self.scatter(we, size=5, color='red'))
        
        #HDSCAN####################################################################################
        hand_mesh, wireframe = self.obj_model.hand_model.get_go_mesh(color='rgb(0,95,115)', opacity=0.3)
        data.append(hand_mesh)
        data.append(wireframe)
        # contact_points = self.obj_model.contact_points
        # contact_normals = self.obj_model.contact_normals
        # labels_n = self.obj_model.labels_n
        # labels_p = self.obj_model.labels_p
        # # print("contact_points", contact_points.shape)
        # # print("contact_normals", contact_normals.shape)
        # # print("labels_n", labels_n.shape)
        # # print("labels_p", labels_p.shape)
        # # print("labels_n", labels_n)
        # # print("labels_p", labels_p)
        # # # colors = ['red', 'blue', 'green', 'yellow', 'purple']     
        # colors = ['rgb(0,18,25)','rgb(0,95,115)','rgb(10,147,150)','rgb(148, 210, 189)','rgb(233, 216, 166)','rgb(238, 155, 0)','rgb(202, 103, 2)','rgb(187, 62, 3)','rgb(174, 32, 18)','rgb(155, 34, 38)']   
        # colors_normal = ['rgb(0, 18, 25)','rgb(202, 103, 2)','rgb(10, 147, 150)']
        # for i, label in enumerate(np.unique(labels_n)):
        #     data.append(self.lines(contact_points[labels_n == label], contact_points[labels_n == label] + contact_normals[labels_n == label]*0.03, color=colors_normal[i],width=7))    
        # color_points = ['rgb(238, 155, 0)','rgb(0, 95, 115)','rgb(187, 62, 3)','rgb(148, 210, 189)','rgb(155, 34, 38)']
        # for i, label in enumerate(np.unique(labels_p)):
        #     data.append(self.scatter(contact_points[labels_p == label], size=10, color=color_points[i]))
        ###############################################################################################
        
        
        # data.append(hand_mesh)
        # print("we", we.size())
        # data.append(self.lines(self.x.repeat(1,4,1).reshape(-1,3).detach().cpu().numpy(), we.reshape(-1,3).detach().cpu().numpy()*0.1+self.x.repeat(1,4,1).reshape(-1,3).detach().cpu().numpy(), color='red'))
        # data.append(self.scatter(we.reshape(-1,3).detach().cpu().numpy(), size=5, color='red'))
        # points = self.scatter(self.x.squeeze(0).detach().cpu().numpy(),size = 10, color='blue')

        # robot_p = self.robot.get_keypoints(self.q).squeeze()
        # robot_p = robot_p[1:, :]
        # print("robot_p", robot_p.size())    
        # r_normed = (robot_p.view(-1,3) - robot_p.view(-1,3).mean(0))/robot_p.view(-1,3).norm(dim=-1).max()
        # t_normed = (self.x.view(-1,3) - self.x.view(-1,3).mean(0))/self.x.view(-1,3).norm(dim=-1).max()



        # data.append(self.scatter(r_normed.detach().cpu().numpy(), size = 10, color='green'))
        # data.append(self.scatter(t_normed.detach().cpu().numpy(), size = 10, color='red'))
        # T,_,i,ids = icp.icp(r_normed.detach().cpu().numpy(), t_normed.detach().cpu().numpy(), max_iterations=100, tolerance=1e-6)
        # T = torch.tensor(T, dtype=torch.float32).to(self.device)
        # print(T)
        # print("ids", ids, "i", i)
        # r_normed_transformed = (T[:3,:3] @ r_normed.T + T[:3,3].reshape(-1,1)).T


        # fingertip = self.obj_model.hand_model.get_fingertips().squeeze().detach().cpu().numpy()
        # x = self.x.squeeze().detach().cpu().numpy()
        # contact = self.obj_model.contact_target.squeeze().detach().cpu().numpy()
        # robot = self.robot.get_keypoints(self.q).squeeze().detach().cpu().numpy()
        
        # data.append(self.scatter(fingertip[0, None], size = 10, color='red', text="M"))
        # data.append(self.scatter(fingertip[1, None], size = 10, color='blue', text="M"))
        # data.append(self.scatter(fingertip[2, None], size = 10, color='green', text="M"))
        # data.append(self.scatter(fingertip[3, None], size = 10, color='yellow', text="M"))
        # data.append(self.scatter(fingertip[4, None], size = 10, color='purple', text="M"))

        # data.append(self.scatter(x[0, None], size = 10, color='red', text="X"))
        # data.append(self.scatter(x[1, None], size = 10, color='blue', text="X"))
        # data.append(self.scatter(x[2, None], size = 10, color='green', text="X"))
        # data.append(self.scatter(x[3, None], size = 10, color='yellow', text="X"))
        # data.append(self.scatter(x[4, None], size = 10, color='purple', text="X"))
        # colors = ['red', 'blue', 'green', 'yellow', 'purple']
        # for i in range(contact.shape[0]):
        #     data.append(self.scatter(contact[i, None], size = 10, color=colors[i], text="C"))
        #     data.append(self.scatter(x[i, None], size = 5, color=colors[i], text="X"))
        # data.append(self.scatter(contact[0, None], size = 10, color='red', text="C"))
        # data.append(self.scatter(contact[1, None], size = 10, color='blue', text="C"))
        # data.append(self.scatter(contact[2, None], size = 10, color='green', text="C"))
        # data.append(self.scatter(contact[3, None], size = 10, color='yellow', text="C"))
        # data.append(self.scatter(contact[4, None], size = 10, color='purple', text="C"))

        # data.append(self.scatter(robot[5, None], size = 10, color='red', text="R"))
        # data.append(self.scatter(robot[1, None], size = 10, color='blue', text="R"))
        # data.append(self.scatter(robot[2, None], size = 10, color='green', text="R"))
        # data.append(self.scatter(robot[3, None], size = 10, color='yellow', text="R"))
        # data.append(self.scatter(robot[4, None], size = 10, color='purple', text="R"))

        # data.append(self.scatter(r_normed_transformed.detach().cpu().numpy(), size = 10, color='blue'))
        # data.append(self.scatter(self.robot_object_contact_check(self.q)[1].squeeze().detach().cpu().numpy(), size = 10, color='red'))
        #points = self.scatter(self.robot.get_keypoints(self.q).squeeze().detach().cpu().numpy()[-1,None])
        #print(self.robot.get_keypoints(self.q).size())
        #print(self.robot.get_keypoints(self.q))
        data.append(self.obj_model.draw(color='gray', opacity=0.5))
        # data.append(points)
        
        # for go_mesh in data:
        #     if isinstance(go_mesh, list):
        #         for track in go_mesh:
        #             x = track.x
        #             y = track.y
        #             z = track.z
                    
        #             track.x = x
        #             track.y = -y
        #             track.z = -z
        theta = np.radians(45)
        fig = go.Figure(data = data)
        camera = dict(
            
            eye=dict(x=1.87, y=0.88, z=0.64), #eye=dict(x=1.87, y=0.88, z=0.64),
            up = dict(x=0, y=0, z=1),
            center = dict(x=0, y=0, z=0)
        )
        fig.update_layout(scene_camera=camera)
        if transparent_background:
            fig.update_layout(
            scene=dict(
                xaxis=dict(showbackground=False,  # Hides the x-axis background
                            tickfont=dict(color='rgba(0,0,0,0)')),  # Transparent x-axis ticks
                yaxis=dict(showbackground=False,  # Hides the y-axis background
                            tickfont=dict(color='rgba(0,0,0,0)')),  # Transparent y-axis ticks
                zaxis=dict(showbackground=False,  # Hides the z-axis background
                            tickfont=dict(color='rgba(0,0,0,0)')),  # Transparent z-axis ticks
            ),
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
            legend=dict(font=dict(color='rgba(0,0,0,0)'))  # Transparent legend
            )
        fig.show()           

    def first_stage_opt(self, test_bar):
        data_fc = self.fc_optimization_verbose(test_bar)
        data_gf = self.gf_optimization_verbose(test_bar)
        if self.robot.robot_model != "Robotiq":
            data_gq = self.gq_optimization_verbose(test_bar)
        #data_gq = self.gq_optimization_verbose(test_bar)
        data_q = self.q_optimization_verbose_v2(test_bar)
        if self.robot.robot_model != "Robotiq":
            data_stage1 = dict(gf_0 = data_gf["Gf"], r_0 = data_gq["r_start"], gf_1 = data_gq["Gf"], r_1 = data_gq["radius"], x = self.x.clone(), w = self.w.clone(), q = self.q.clone())
        else:
            data_stage1 = dict(gf_0 = data_gf["Gf"], x = self.x.clone(), w = self.w.clone(), q = self.q.clone())
        return data_stage1
    
    def second_stage_opt(self, test_bar):
        # self.robot_contact, self.actual_contact = self.robot_object_contact_check(self.q)
        # self.obj_model.contact_target = self.actual_contact
        data_fcre = self.fc_optimization_verbose(test_bar)
        data_gfre = self.gf_optimization_verbose(test_bar)
        if self.robot.robot_model != "Robotiq":
            data_gqre = self.gq_optimization_verbose(test_bar)
        # data_gqre = self.gq_optimization_verbose(test_bar)
        data_qre = self.q_optimization_verbose_v2(test_bar)
        self.robot_contact, self.actual_contact = self.robot_object_contact_check(self.q)
        if self.robot.robot_model != "Robotiq":
            data_stage2 = dict(gf_0r = data_gfre["Gf"], r_0r = data_gqre["r_start"], gf_1r = data_gqre["Gf"], r_1r = data_gqre["radius"], 
                            xr = self.x.clone(), wr = self.w.clone(), qr = self.q.clone(),
                            actual_contact = self.actual_contact, robot_contact = self.robot_contact)
        else:
            data_stage2 = dict(gf_0r = data_gfre["Gf"], xr = self.x.clone(), wr = self.w.clone(), qr = self.q.clone(),
                            actual_contact = self.actual_contact, robot_contact = self.robot_contact)
        return data_stage2
    
    def second_stage_opt_v2(self, test_bar):
        self.robot_contact, self.actual_contact = self.robot_object_contact_check(self.q)
        self.obj_model.contact_target = self.actual_contact
        data_gfre = self.gf_reoptimization_verbose(test_bar)
        if self.robot.robot_model != "Robotiq":
            data_gqre = self.gq_reoptimization_verbose(test_bar)
            data_stage2 = dict(gf_0r = data_gfre["Gf"], r_0r = data_gqre["r_start"], gf_1r = data_gqre["Gf"], r_1r = data_gqre["radius"], 
                            xr = self.x.clone(), wr = self.w.clone(), qr = self.q.clone(),
                            actual_contact = self.actual_contact, robot_contact = self.robot_contact)
        else:
            data_stage2 = dict(gf_0r = data_gfre["Gf"], xr = self.x.clone(), wr = self.w.clone(), qr = self.q.clone(),
                            actual_contact = self.actual_contact, robot_contact = self.robot_contact)
        return data_stage2

    def first_stage_opt_wo_gq(self, test_bar):
        start_time = time.time()
        data_fc = self.fc_optimization_verbose(test_bar)
        fc_time = time.time() - start_time
        data_q = self.q_optimization_verbose_v2(test_bar)
        q_time = time.time() - start_time - fc_time
        
        total_time = time.time() - start_time
        data_stage1 = dict(gf_0 = data_fc["Gf"], sdf=data_fc["sdf"], x = self.x.clone(), w = self.w.clone(), q = self.q.clone(), fc_time = fc_time, q_time = q_time, total_time = total_time, distance_loss = data_q["distance_loss"])
        return data_stage1
    
    def second_stage_opt_wo_gq(self, test_bar):
        self.robot_contact, self.actual_contact = self.robot_object_contact_check(self.q)
        # self.obj_model.contact_target = self.actual_contact
        
        # data_gfre = self.gf_reoptimization_verbose(test_bar)
        data_stage2 = dict(xr = self.x.clone(), wr = self.w.clone(), qr = self.q.clone(),
                           actual_contact = self.actual_contact, robot_contact = self.robot_contact)
        return data_stage2

    def second_stage_opt_wo_gq_refine(self, test_bar):
        self.robot_contact, self.actual_contact = self.robot_object_contact_check(self.q)
        # if self.refine_flag:
        self.obj_model.contact_target = self.actual_contact
        start_time = time.time()
        data_fc = self.fc_optimization_verbose(test_bar)
        fc_time = time.time() - start_time
        data_q = self.q_optimization_verbose_v2(test_bar)
        q_time = time.time() - start_time - fc_time
        total_time = time.time() - start_time
        # data_gfre = self.gf_reoptimization_verbose(test_bar)
        data_stage2 = dict(xr = self.x.clone(), wr = self.w.clone(), qr = self.q.clone(),
                        actual_contact = self.actual_contact, robot_contact = self.robot_contact, fc_time_r = fc_time, q_time_r = q_time, total_time_r = total_time, distance_loss_r = data_q["distance_loss"])
        # else:
        #     data_stage2 = dict(xr = self.x.clone(), wr = self.w.clone(), qr = self.q.clone(),
        #                    actual_contact = self.actual_contact, robot_contact = self.robot_contact)

        return data_stage2
    
    def first_stage_opt_nv(self, test_bar):
        start_time = time.time()
        data_q = self.q_optimization_verbose_nv(test_bar)
        q_time = time.time() - start_time

        data_stage_nv = dict(q = self.q.clone(), q_time = q_time)
        return data_stage_nv
    
    def run_idx_json_nv(self, idx):
        test_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), smoothing=0.9)
        input_pack, obj_pack, mano_pack = self.dataset[idx]
        obj_mesh_v = obj_pack["verts"].to(self.device).unsqueeze(0)
        obj_mesh_f = obj_pack["faces"].to(self.device).unsqueeze(0)
        obj_certre = obj_pack["centre"].to(self.device).unsqueeze(0)
        hand_mesh_v = mano_pack["verts"].to(self.device).unsqueeze(0)
        hand_mesh_f = mano_pack["faces"].to(self.device).unsqueeze(0)
        joints = mano_pack["joint"].to(self.device).unsqueeze(0)
        hand_transformation = mano_pack["transformation"].to(self.device).unsqueeze(0)
        try:
            self.initialization(obj_mesh_v, obj_mesh_f, obj_certre, hand_mesh_v, hand_mesh_f, joints, hand_transformation)
            print("target", self.target)
            data_nv = self.first_stage_opt_nv(test_bar)
            file_name = str(idx)+"_"+self.robot.robot_model + "_" + "0_"+ "mu_%.1f" %self.obj_model.loss.mu.item()
            data_nv["data_sample_config"] = self.dataset.data_sample
        except:
            traceback.print_exc()
            file_name = str(idx)+"_"+self.robot.robot_model + "_" + "0_"+ "mu_%.1f" %self.obj_model.loss.mu.item()
            data_nv = {"error": "error", "data_sample_config": self.dataset.data_sample, "idx":idx}
        
        self.write_json(data_nv, file_name)
        # self.draw()
        
    def run_idx_json(self, idx, draw = False, save = False, frontend = False):
        test_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), smoothing=0.9)
        input_pack, obj_pack, mano_pack = self.dataset[idx]
        # change to estimated object mesh
        if frontend:
            mesh_name = input_pack["file_name"] + "_obj.ply"
            mesh_dir = osp.join(osp.dirname(__file__), "CtcSDF_v2", "hmano_osdf", "mesh", mesh_name)
            mesh = tm.load(mesh_dir, process=False)
            hand_trans = obj_pack["hand_trans"].to(self.device).unsqueeze(0)
            obj_mesh_v = torch.tensor(mesh.vertices, dtype=torch.float32).to(self.device).unsqueeze(0)
            # obj_mesh_v = obj_mesh_v - hand_trans
            obj_mesh_f = torch.tensor(mesh.faces, dtype=torch.int32).to(self.device).unsqueeze(0)
            obj_centre = torch.tensor(mesh.center_mass, dtype=torch.float32).to(self.device).unsqueeze(0)
            obj_mesh_v = obj_mesh_v - hand_trans
            obj_centre = obj_centre - hand_trans
            hand_mesh_name = input_pack["file_name"] + "_hand.ply"
            hand_mesh_dir = osp.join(osp.dirname(__file__), "CtcSDF_v2", "hmano_osdf", "mesh_hand", hand_mesh_name)
            hand_mesh = tm.load(hand_mesh_dir, process=False)
            hand_mesh_v = torch.tensor(hand_mesh.vertices, dtype=torch.float32).to(self.device).unsqueeze(0)
            hand_mesh_f = torch.tensor(hand_mesh.faces, dtype=torch.int32).to(self.device).unsqueeze(0)
            hand_mesh_v = hand_mesh_v - hand_trans
        else:
            obj_mesh_v = obj_pack["verts"].to(self.device).unsqueeze(0)
            obj_mesh_f = obj_pack["faces"].to(self.device).unsqueeze(0)
            obj_centre = obj_pack["centre"].to(self.device).unsqueeze(0)
            hand_mesh_v = mano_pack["verts"].to(self.device).unsqueeze(0)
            hand_mesh_f = mano_pack["faces"].to(self.device).unsqueeze(0)
        joints = mano_pack["joint"].to(self.device).unsqueeze(0)
        hand_transformation = mano_pack["transformation"].to(self.device).unsqueeze(0)
        try:
            self.initialization(obj_mesh_v, obj_mesh_f, obj_centre, hand_mesh_v, hand_mesh_f, joints, hand_transformation)
            data_stage1 = self.first_stage_opt_wo_gq(test_bar)
            data_stage2 = self.second_stage_opt_wo_gq(test_bar)
            file_name = str(idx)+"_"+self.robot.robot_model + "_" + "0_"+ "mu_%.1f" %self.obj_model.loss.mu.item()
            data_stage1.update(data_stage2)
            data_stage1["data_sample_config"] = self.dataset.data_sample
        except Exception as e:
            traceback.print_exc()
            print(e)
            file_name = str(idx)+"_"+self.robot.robot_model + "_" + "0_"+ "mu_%.1f" %self.obj_model.loss.mu.item()
            data_stage1 = {"error": str(e), "data_sample_config": self.dataset.data_sample, "idx": idx}
        if save:
            # self.result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
            self.write_json(data_stage1, file_name)
        if draw:
            self.draw()
        # self.draw()
        
    def run_wo_gq(self, restart = None, indices = None):
        if restart is not None:
            sampler = SubsetRestartSampler(restart, self.dataset.__len__())
            initial = restart
        if indices is not None:
            sampler = TensorIndexSampler(indices)
            initial = 0

        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=4, sampler=sampler)

        test_bar = tqdm(enumerate(self.dataloader), initial= initial, total=len(self.dataloader), smoothing=0.9)
        for i, (input_pack, obj_pack, mano_pack) in test_bar:
            if restart:
                i += restart
            if i > 20:
                break
            for j in range(self.repeat):
                obj_mesh_v = obj_pack["verts"].to(self.device)
                obj_mesh_f = obj_pack["faces"].to(self.device)
                obj_certre = obj_pack["centre"].to(self.device)
                hand_mesh_v = mano_pack["verts"].to(self.device)
                hand_mesh_f = mano_pack["faces"].to(self.device)
                joints = mano_pack["joint"].to(self.device)
                hand_transformation = mano_pack["transformation"].to(self.device)
                try:
                    self.initialization(obj_mesh_v, obj_mesh_f, obj_certre, hand_mesh_v, hand_mesh_f, joints, hand_transformation)
                    self.refine_stage = False
                    self.refine_flag = True
                    data_stage1 = self.first_stage_opt_wo_gq(test_bar)
                    self.row_ind, self.col_ind = None, None
                    self.refine_stage = True
                    data_stage2 = self.second_stage_opt_wo_gq_refine(test_bar)
                    if indices is not None:
                        file_name = str(indices[i].item())+"_"+self.robot.robot_model+"_"+str(j) + "_" + "mu_%.1f" %self.obj_model.loss.mu.item()
                    else:
                        file_name = str(i)+"_"+self.robot.robot_model+"_"+str(j) + "_" + "mu_%.1f" %self.obj_model.loss.mu.item()
                    data_stage1.update(data_stage2)
                    data_stage1["data_sample_config"] = self.dataset.data_sample 
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    if indices is not None:
                        file_name = str(indices[i].item())+"_"+self.robot.robot_model+"_"+str(j) + "_" + "mu_%.1f" %self.obj_model.loss.mu.item()
                    else:
                        file_name = str(i)+"_"+self.robot.robot_model+"_"+str(j) + "_" + "mu_%.1f" %self.obj_model.loss.mu.item()                    
                    data_stage1 = {"error": str(e), "data_sample_config": self.dataset.data_sample, "idx": i}
                    # continue
                print(file_name)
                self.write_json(data_stage1, file_name)

    def run_wo_gq_nv(self, restart = 0):
        if restart != 0:
            sampler = SubsetRestartSampler(restart, self.dataset.__len__())
            self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=1, sampler=sampler)

        test_bar = tqdm(enumerate(self.dataloader), initial= restart, total=len(self.dataloader), smoothing=0.9)
        for i, (input_pack, obj_pack, mano_pack) in test_bar:
            i += restart
            for j in range(self.repeat):
                obj_mesh_v = obj_pack["verts"].to(self.device)
                obj_mesh_f = obj_pack["faces"].to(self.device)
                obj_certre = obj_pack["centre"].to(self.device)
                hand_mesh_v = mano_pack["verts"].to(self.device)
                hand_mesh_f = mano_pack["faces"].to(self.device)
                joints = mano_pack["joint"].to(self.device)
                hand_transformation = mano_pack["transformation"].to(self.device)
                try:
                    self.initialization(obj_mesh_v, obj_mesh_f, obj_certre, hand_mesh_v, hand_mesh_f, joints, hand_transformation)
                    data_nv = self.first_stage_opt_nv(test_bar)

                    file_name = str(i)+"_"+self.robot.robot_model+"_"+str(j) + "_" + "mu_%.1f" %self.obj_model.loss.mu.item()
                    data_nv["data_sample_config"] = self.dataset.data_sample
                except Exception as e:
                    traceback.print_exc()
                    file_name = str(i)+"_"+self.robot.robot_model+"_"+str(j) + "_" + "mu_%.1f" %self.obj_model.loss.mu.item()
                    data_nv = {"error": str(e), "data_sample_config": self.dataset.data_sample, "idx": i}
                    # continue
                
                self.write_json(data_nv, file_name)
                
    def run(self):
        test_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), smoothing=0.9)
        for i, (input_pack, obj_pack, mano_pack) in test_bar:
            for j in range(self.repeat):
                obj_mesh_v = obj_pack["verts"].to(self.device)
                obj_mesh_f = obj_pack["faces"].to(self.device)
                obj_certre = obj_pack["centre"].to(self.device)
                hand_mesh_v = mano_pack["verts"].to(self.device)
                hand_mesh_f = mano_pack["faces"].to(self.device)
                joints = mano_pack["joint"].to(self.device)
                hand_transformation = mano_pack["transformation"].to(self.device)
                self.initialization(obj_mesh_v, obj_mesh_f, obj_certre, hand_mesh_v, hand_mesh_f, joints, hand_transformation)
                
                data_stage1 = self.first_stage_opt(test_bar)
                data_stage2 = self.second_stage_opt_v2(test_bar)
                file_name = str(i)+"_"+self.robot.robot_model+"_"+str(j) + "_" + "mu_%.1f" %self.obj_model.loss.mu.item()
                data_stage1.update(data_stage2)
                data_stage1["data_sample_config"] = self.dataset.data_sample 
                self.write_json(data_stage1, file_name)
                
    def run_w_idx(self, idx):
        test_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), smoothing=0.9)
        input_pack, obj_pack, mano_pack = self.dataset[idx]
        color = input_pack['color_img']

        obj_mesh_v = obj_pack["verts"].to(self.device).unsqueeze(0)
        obj_mesh_f = obj_pack["faces"].to(self.device).unsqueeze(0)
        obj_certre = obj_pack["centre"].to(self.device).unsqueeze(0)
        hand_mesh_v = mano_pack["verts"].to(self.device).unsqueeze(0)
        hand_mesh_f = mano_pack["faces"].to(self.device).unsqueeze(0)
        joints = mano_pack["joint"].to(self.device).unsqueeze(0)
        hand_transformation = mano_pack["transformation"].to(self.device).unsqueeze(0)
        self.initialization(obj_mesh_v, obj_mesh_f, obj_certre, hand_mesh_v, hand_mesh_f, joints, hand_transformation)
        
        self.fc_optimization_verbose(test_bar)
        # self.gf_optimization_verbose(test_bar)
        # self.gq_optimization_verbose(test_bar)
        self.q_optimization_verbose_v2(test_bar)

        self.robot_contact, self.actual_contact = self.robot_object_contact_check(self.q)
        self.obj_model.contact_target = self.actual_contact
        # self.fc_optimization_verbose(test_bar)
        # self.gf_optimization_verbose(test_bar)

        # self.gq_reoptimization_verbose(test_bar)

        if self.visualize:
            # from CtcSDF.CtcViz import data_viz
            # data_viz('color',color.squeeze(0).numpy())
            self.draw()
    
    def run_w_idx_nv(self, idx):
        test_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), smoothing=0.9)
        input_pack, obj_pack, mano_pack = self.dataset[idx]
        color = input_pack['color_img']

        obj_mesh_v = obj_pack["verts"].to(self.device).unsqueeze(0)
        obj_mesh_f = obj_pack["faces"].to(self.device).unsqueeze(0)
        obj_certre = obj_pack["centre"].to(self.device).unsqueeze(0)
        hand_mesh_v = mano_pack["verts"].to(self.device).unsqueeze(0)
        hand_mesh_f = mano_pack["faces"].to(self.device).unsqueeze(0)
        joints = mano_pack["joint"].to(self.device).unsqueeze(0)
        hand_transformation = mano_pack["transformation"].to(self.device).unsqueeze(0)
        self.initialization(obj_mesh_v, obj_mesh_f, obj_certre, hand_mesh_v, hand_mesh_f, joints, hand_transformation)
        
        data_nv = self.first_stage_opt_nv(test_bar)

        self.robot_contact, self.actual_contact = self.robot_object_contact_check(self.q)
        self.obj_model.contact_target = self.actual_contact



        # self.fc_optimization_verbose(test_bar)
        # self.gf_optimization_verbose(test_bar)

        # self.gq_reoptimization_verbose(test_bar)

        # if self.visualize:
        #     # from CtcSDF.CtcViz import data_viz
        #     # data_viz('color',color.squeeze(0).numpy())
        #     self.draw()
    
    def run_verbose(self):
        test_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), smoothing=0.9)
        #for i, (input_pack, obj_pack, mano_pack) in test_bar:
        input_pack, obj_pack, mano_pack = next(iter(self.dataloader))
        obj_mesh_v = obj_pack["verts"].to(self.device)
        obj_mesh_f = obj_pack["faces"].to(self.device)
        obj_certre = obj_pack["centre"].to(self.device)
        hand_mesh_v = mano_pack["verts"].to(self.device)
        hand_mesh_f = mano_pack["faces"].to(self.device)
        joints = mano_pack["joint"].to(self.device)
        hand_transformation = mano_pack["transformation"].to(self.device)
        self.initialization(obj_mesh_v, obj_mesh_f, obj_certre, hand_mesh_v, hand_mesh_f, joints, hand_transformation)
        self.fc_optimization_verbose(test_bar)
        self.gq_optimization_verbose(test_bar)
        self.q_optimization_verbose(test_bar)
        #print("hand contact", self.obj_model.hand_contact)
        if self.visualize:
            self.draw()
            
    def run_iter(self):
        test_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), smoothing=0.9)
        for i, (input_pack, obj_pack, mano_pack) in test_bar:
            obj_mesh_v = obj_pack["verts"].to(self.device)
            obj_mesh_f = obj_pack["faces"].to(self.device)
            obj_certre = obj_pack["centre"].to(self.device)
            hand_mesh_v = mano_pack["verts"].to(self.device)
            hand_mesh_f = mano_pack["faces"].to(self.device)
            joints = mano_pack["joint"].to(self.device)
            hand_transformation = mano_pack["transformation"].to(self.device)
            self.initialization(obj_mesh_v, obj_mesh_f, obj_certre, hand_mesh_v, hand_mesh_f, joints, hand_transformation)
            self.fc_optimization_verbose(test_bar)
            self.gq_optimization_verbose(test_bar)
            self.q_optimization_verbose(test_bar)
            #with open
            
    def run_viz(self, idx):
        #test_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), smoothing=0.9)
        input_pack, obj_pack, mano_pack = self.dataset[idx]
        obj_mesh_v = obj_pack["verts"].to(self.device).unsqueeze(0)
        obj_mesh_f = obj_pack["faces"].to(self.device).unsqueeze(0)
        obj_certre = obj_pack["centre"].to(self.device).unsqueeze(0)
        hand_mesh_v = mano_pack["verts"].to(self.device).unsqueeze(0)
        hand_mesh_f = mano_pack["faces"].to(self.device).unsqueeze(0)
        joints = mano_pack["joint"].to(self.device).unsqueeze(0)
        hand_transformation = mano_pack["transformation"].to(self.device).unsqueeze(0)
        self.initialization(obj_mesh_v, obj_mesh_f, obj_certre, hand_mesh_v, hand_mesh_f, joints, hand_transformation)
        contact_points = self.obj_model.contact_target
        data = []
        rob_points, rob_normals = self.robot.get_surface_points_normal_updated(self.q)
        rob_cot_points, rob_cot_normals = self.robot.get_contact_points_normal_updated(self.q)
        key_points = self.robot.get_keypoints(self.q)
        # data.append(self.lines(rob_cot_points.view(-1,3), rob_cot_points.view(-1,3) + rob_cot_normals.view(-1,3) * 0.01))
        #data.append(self.scatter(rob_cot_points.squeeze(0).view(-1,3).detach().cpu().numpy(), size=10, color='red'))
        # data.extend(self.robot.get_mesh_updated(q =self.q, opacity=0.9))
        # data.append(self.scatter(key_points.squeeze().detach().cpu().numpy()[-1,None], size = 10, color='red'))
        #hand = self.mesh(hand_mesh_v.squeeze(0).cpu(), hand_mesh_f.squeeze(0).cpu(), opacity=0.5)
        #print("size", contact_points[0,:].size())

        #points = self.scatter(contact_points[0,:].unsqueeze(0).detach().cpu().numpy(), size = 10, color='blue')
        # data.append(self.scatter(contact_points[1:].squeeze(0).detach().cpu().numpy(), size = 10, color='red'))
        # data.append(self.scatter(contact_points[0:].squeeze(0).detach().cpu().numpy(), size = 10, color='blue'))
        data.append(self.obj_model.draw(color="gray", opacity=0.3))
        # data.append(points)
        data.append(self.obj_model.hand_model.get_go_mesh(color="white", opacity=0.3))
        # data.append(self.scatter(self.obj_model.contact_0.squeeze(0).cpu().numpy(), 4, 'red'))  
        data.append(self.scatter(self.obj_model.contact_1.squeeze(0).cpu().numpy(), 4, 'green'))  
        # data.append(self.lines(self.obj_model.contact_0, self.obj_model.contact_0_normal * 0.03+ self.obj_model.contact_0, color='red', width=4))
        data.append(self.lines(self.obj_model.contact_1, self.obj_model.contact_1_normal * 0.03+ self.obj_model.contact_1, color='green', width=4))
        
        data.append(self.scatter(self.obj_model.contact_0.squeeze(0).cpu().numpy(), 4, 'red'))
        data.append(self.lines(self.obj_model.contact_0, self.obj_model.contact_0_normal * 0.03+ self.obj_model.contact_0, color='red', width=4))

        data.append(self.scatter(self.obj_model.contact_target.squeeze(0).cpu().numpy(), 8, 'blue'))
        
        color = ['red', 'blue', 'yellow', 'purple']
        
        # for i,subset in enumerate(self.obj_model.sub_contact):
        #     data.append(self.scatter(subset.squeeze(0).cpu().numpy(), 4, color=color[i]))
        #     data.append(self.lines(subset, self.obj_model.sub_contact_normal[i] * 0.03+ subset, color=color[i], width=4))
            
        
        # data.append(self.scatter(self.obj_model.hand_contact_points().squeeze(0).detach().cpu().numpy(), size = 4, color='red'))
        # data.append(self.scatter(self.obj_model.hand_contact_objnormals().squeeze(0).detach().cpu().numpy()*0.1, size = 4, color='green'))
        # data.append(self.lines(self.obj_model.hand_contact_points(), self.obj_model.hand_contact_objnormals() * 0.01+ self.obj_model.hand_contact_points() ))
        
        # data.append(self.scatter(self.obj_model.hand_contact_points().squeeze(0).detach().cpu().numpy(), size = 4, color='red'))
        
        self.go_graph(data, transparent_background=True)
        
    def lines(self, start, end, color='red', width=2):
        N, _ = start.shape
        x1, y1, z1 = start[:,0], start[:,1], start[:,2]
        x2, y2, z2 = end[:,0], end[:,1], end[:,2]
        x_lines = []
        y_lines = []
        z_lines = []
        for i in range(N):
            x_lines.extend([x1[i], x2[i], None])
            y_lines.extend([y1[i], y2[i], None])
            z_lines.extend([z1[i], z2[i], None])
        lines = go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode='lines',
            line=dict(color=color, width=width)
        )
        return lines
        
    def mesh(self, mesh_v, mesh_f, color = 'white', opacity = 0.5):
        mesh = go.Mesh3d(
            x=mesh_v[:, 0],
            y=mesh_v[:, 1],
            z=mesh_v[:, 2],
            i=mesh_f[:, 0],
            j=mesh_f[:, 1],
            k=mesh_f[:, 2],
            color=color,
            opacity=opacity
        )
        return mesh
        
    def scatter(self, points, size, color='red'):
        scatter = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=size,  # Adjust marker size here
                color=color  # Adjust marker color here
            )
        )
        return scatter
    
    def scatter(self, points, size, color='red', text=None):
        scatter = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers+text',
            text=text,
            textposition="top center",
            marker=dict(
                size=size,  # Adjust marker size here
                color=color  # Adjust marker color here
            )
        )
        return scatter
    
    def pyramids(self, base_vertices, apex_vertex, color='red', opacity=0.5):
        assert base_vertices.shape == (4, 3), "Base vertices must be a (4,3) NumPy array"
        assert apex_vertex.shape == (3,), "Apex vertex must be a (3,) NumPy array"

        # Extract x, y, z coordinates
        x = np.append(base_vertices[:, 0], apex_vertex[0])
        y = np.append(base_vertices[:, 1], apex_vertex[1])
        z = np.append(base_vertices[:, 2], apex_vertex[2])
        faces = dict(
        i=[0, 4, 2, 3, 0],  # First vertex of each triangle
        j=[1, 2, 1, 0, 4],  # Second vertex
        k=[2, 3, 4, 4, 1]   # Third vertex
        )
        
        pyramid = go.Mesh3d(x=x, y=y, z=z, opacity=opacity, color=color, i=faces['i'], j=faces['j'], k=faces['k'])
        return pyramid
        
    def go_graph(self, 
                 data: List[go.Scatter3d],
                 transparent_background: bool = False):
        fig = go.Figure(data = data)
        camera = dict(
            eye=dict(x=1.87, y=0.88, z=0.64),
            up = dict(x=0, y=0, z=1),
            center = dict(x=0, y=0, z=0)
        )
        fig.update_layout(scene_camera=camera)
        if transparent_background:
            fig.update_layout(
            scene=dict(
                xaxis=dict(showbackground=False,  # Hides the x-axis background
                            tickfont=dict(color='rgba(0,0,0,0)')),  # Transparent x-axis ticks
                yaxis=dict(showbackground=False,  # Hides the y-axis background
                            tickfont=dict(color='rgba(0,0,0,0)')),  # Transparent y-axis ticks
                zaxis=dict(showbackground=False,  # Hides the z-axis background
                            tickfont=dict(color='rgba(0,0,0,0)')),  # Transparent z-axis ticks
            ),
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
            legend=dict(font=dict(color='rgba(0,0,0,0)'))  # Transparent legend
            )
        fig.show() 
                
def plot_idx(idx, robot):
    task = "GH"
    
    maximum_iter=[7000, 1, 1000]
    dataset = dexycb_trainfullfeed()
    opt = Optimization(robot=robot, dataset=dataset, device="cuda:0", maximum_iter=maximum_iter, visualize=True, repeat=1, mu=0.9, task=task)
    opt.run_idx_json(idx, draw=True, save=True)

def index_gh(idx, robot):

    task = "GH"
    maximum_iter=[7000, 1, 1000]
    dataset = dexycb_testfullfeed(load_mesh=True, pc_sample=1024, data_sample=7, precept=True)
    opt = Optimization(robot=robot, dataset=dataset, device="cpu", maximum_iter=maximum_iter, visualize=True, repeat=1, mu=0.9, task=task, source="sdf")
    # opt.result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'results_sdf_demo')

    opt.run_idx_json(idx, False, True, False)

def run_optimization_nv(robot, task):

    datasample = 20
    dataset = dexycb_testfullfeed(load_mesh=True, pc_sample=1024, data_sample=datasample)
    maximum_iter=[7000, 1, 1000]
    # indices = torch.tensor(list(range(90, 100)), dtype=torch.long)
    for mu in [0.1,0.3,0.5,0.7,0.9]:
        opt = Optimization(robot=robot, dataset=dataset, device="cuda:0", maximum_iter=maximum_iter, visualize=True, repeat=1, mu=mu, task=task)
        opt.run_wo_gq_nv(restart=0)

def run_optimization(robot, task):

    datasample = 20
    dataset = dexycb_testfullfeed(load_mesh=True, pc_sample=1024, data_sample=datasample)
    maximum_iter=[7000, 1, 1000]
    # indices = torch.tensor(list(range(90, 100)), dtype=torch.long)
    for mu in [0.1,0.3,0.5,0.7,0.9]:
        opt = Optimization(robot=robot, dataset=dataset, device="cuda:0", maximum_iter=maximum_iter, visualize=True, repeat=1, mu=mu, task=task)
        opt.run_wo_gq(restart=0)

def optimization_worker(mu, robot_type, dataset, task, maximum_iter, source, device):
    # Initialize the robot inside the worker function
    if robot_type == "Shadow":
        robot = Shadow(batch=1, device=device)
    elif robot_type == "Allegro":
        robot = Allegro(batch=1, device=device)
    elif robot_type == "Barrett":
        robot = Barrett(batch=1, device=device)
    elif robot_type == "Robotiq":
        robot = Robotiq(batch=1, device=device)
    else:
        raise ValueError("Unknown robot type")

    # Ensure the robot and other necessary components are moved to the correct device
    robot.to(device)
    opt = Optimization(robot=robot, dataset=dataset, device=device, maximum_iter=maximum_iter, visualize=True, repeat=1, mu=mu, task=task, source=source)
    if task == "NV":
        opt.run_wo_gq_nv(restart=0)
    else:
        opt.run_wo_gq(restart=0)


def run_optimization_mp(robot1, robot2, task1, task2):
    # Prepare the data sample and dataset
    datasample = 20
    dataset = dexycb_testfullfeed(load_mesh=True, pc_sample=1024, data_sample=datasample)
    maximum_iter = [7000, 1, 1000]

    # List of mu values
    mu_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Use multiprocessing to run optimization for each mu in parallel for two robots
    processes = []
    source = "ycb"
    # Set the start method to 'spawn' to ensure that CUDA contexts are initialized correctly in each process
    mp.set_start_method('spawn', force=True)

    # Run for robot 1 on GPU 0
    for mu in mu_values:
        p = mp.Process(target=optimization_worker, args=(mu, robot1, dataset, task1, maximum_iter, source, "cuda:0"))
        processes.append(p)
        p.start()

    # Run for robot 2 on GPU 1
    for mu in mu_values:
        p = mp.Process(target=optimization_worker, args=(mu, robot2, dataset, task2, maximum_iter, source, "cuda:1"))
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

def optimization_worker_2(mu, robot_type, dataset, task, maximum_iter, source, device):
    # Initialize the robot inside the worker function
    if robot_type == "Shadow":
        robot = Shadow(batch=1, device=device)
    elif robot_type == "Allegro":
        robot = Allegro(batch=1, device=device)
    elif robot_type == "Barrett":
        robot = Barrett(batch=1, device=device)
    elif robot_type == "Robotiq":
        robot = Robotiq(batch=1, device=device)
    else:
        raise ValueError("Unknown robot type")


    # Initialize the optimization process with the robot now properly set to its device
    opt = Optimization(robot=robot, dataset=dataset, device=device, maximum_iter=maximum_iter, visualize=True, repeat=1, mu=mu, task=task, source=source)
    if task == "NV":
        opt.run_wo_gq_nv(restart=0)
    else:
        opt.run_wo_gq(restart=0)

def run_optimization_mp_2():
    # Define tasks, mu values, and source
    robots = ["Allegro","Shadow","Barrett","Robotiq"]  # List robot types as strings
    tasks = ["GH"]
    mu_values = [0.1]
    source = "ycb"
    datasample = 20
    dataset = dexycb_testfullfeed(load_mesh=True, pc_sample=1024, data_sample=datasample, precept=True if source == "sdf" else False)
    
    # Generate task arguments combining robot types, mu values, and tasks
    task_args = list(itertools.product(robots, mu_values, tasks))

    # Device assignment and multiprocessing setup
    num_gpus = 1
    current_gpu = 0
    processes = []
    mp.set_start_method('spawn')  # Ensures each process has its own CUDA context

    for robot_type, mu, task in task_args:
        device = f"cuda:{current_gpu}"
        p = mp.Process(target=optimization_worker_2, args=(mu, robot_type, dataset, task, [7000,1,1000], source, device))
        processes.append(p)
        p.start()

        # Round-robin scheduling on GPUs
        current_gpu = (current_gpu + 1) % num_gpus
        if len(processes) == num_gpus:
            for p in processes:
                p.join()
            processes = []  # Reset for the next batch

if __name__ == "__main__":
    run_optimization_mp_2()
