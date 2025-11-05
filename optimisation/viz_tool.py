import numpy as np
import os
osp = os.path
import sys
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), "CtcSim", "bullet"))
from CtcSim.bullet.PB_test import ycb_opt_fetcher
from optimisation_tmp.CtcObj import object_sdf
import matplotlib.pyplot as plt
from optimisation_tmp.robot import Shadow, Allegro, Barrett, Robotiq
import json
import torch
import re
from torch.utils.data import Dataset
from dex_ycb_toolkit.dex_ycb import DexYCBDataset
import trimesh as tm
from manopth.manolayer import ManoLayer
from torch.utils.data import DataLoader
import pytorch_volumetric as pv
import traceback
from CtcSDF.CtcViz import data_viz
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm
from scipy.spatial import cKDTree as KDTree
from scipy.optimize import linear_sum_assignment

import scipy.stats as stats
import optimisation_tmp.mathBox as mb
"""
obj categroy {
1: '002_master_chef_can', 
2: '003_cracker_box', 
3: '004_sugar_box', 
4: '005_tomato_soup_can', 
5: '006_mustard_bottle',
6: '007_tuna_fish_can', 
7: '008_pudding_box', 
8: '009_gelatin_box', 
9: '010_potted_meat_can', 
10: '011_banana', 
11: '019_pitcher_base', 
12: '021_bleach_cleanser', 
13: '024_bowl', 
14: '025_mug', 
15: '035_power_drill', 
16: '036_wood_block', 
17: '037_scissors', 
18: '040_large_marker', 
19: '051_large_clamp', 
20: '052_extra_large_clamp', 
21: '061_foam_brick'}
"""

# 1. GF over mu for every robot --OPT
# 2. GF over mu for every object --OPT

# 3. computing time on force closure for every object --OPT
# 4. computing time on kinematics for every robot --OPT

# 5. success over mu for every robot --SIM
# 6. success over mu for every object --SIM
# 7. success over mu for every robot with nv --SIM
class result_loader:
    def __init__(self, exp, task, source='gt'):
        self.task = exp    
        if task == 'opt':
            if source == 'gt':
                if exp == 'gh':
                    self.result_path = osp.join(osp.dirname(osp.abspath(__file__)),'results')
                else:
                    self.result_path = osp.join(osp.dirname(osp.abspath(__file__)),'results_nv')
            if source == 'sdf':
                if exp == 'gh':
                    self.result_path = osp.join(osp.dirname(osp.abspath(__file__)),'results_sdf')
                else:
                    self.result_path = osp.join(osp.dirname(osp.abspath(__file__)),'results_nv_sdf')
        if task == 'sim':
            if source == 'gt':
                if exp == 'gh':
                    self.result_path = osp.join(osp.dirname(osp.abspath(__file__)),'CtcSim','bullet', 'sim_result')
                else:
                    self.result_path = osp.join(osp.dirname(osp.abspath(__file__)),'CtcSim','bullet', 'sim_result_nv')
            if source == 'sdf':
                if exp == 'gh':
                    self.result_path = osp.join(osp.dirname(osp.abspath(__file__)),'CtcSim','bullet', 'sim_result_sdf')
                else:
                    self.result_path = osp.join(osp.dirname(osp.abspath(__file__)),'CtcSim','bullet', 'sim_result_sdf_nv')
        self.obj_list = {0: 'master_chef_can',
                         1: 'cracker_box',
                         2: 'sugar_box',
                         3: 'tomato_soup_can',
                         4: 'mustard_bottle',
                         5: 'tuna_fish_can',
                         6: 'pudding_box',
                         7: 'gelatin_box',
                         8: 'potted_meat_can',
                         9: 'banana',
                         10: 'pitcher_base',
                         11: 'bleach_cleanser',
                         12: 'bowl',
                         13: 'mug',
                         14: 'power_drill',
                         15: 'wood_block',
                         16: 'scissors',
                         17: 'large_marker',
                         #18: '051_large_clamp',
                         18: 'extra_large_clamp',
                         19: 'foam_brick'}

    def load_SR_obj(self):
        def extract_middle_integer_sim(filename):
            match = re.search(r'_([0-9]+)\.json$', filename)  # Find number before ".txt"
            return int(match.group(1)) if match else float('inf')  # If no match, put at end
   
        SR = {}
        for robot in ['Shadow', 'Allegro', 'Barrett', 'Robotiq']:
            for obj in self.obj_list.values():
                SR[(robot, obj)] = []
        for robot in ['Shadow', 'Allegro', 'Barrett', 'Robotiq']:
            if robot == 'Shadow':
                sample = 20
            else:
                sample = 20
            unsort_file_list = [f for f in os.listdir(self.result_path) if robot in f and ('0.9' in f or '0.7' in f or '0.5' in f)]
            file_list = sorted(unsort_file_list, key=extract_middle_integer_sim)
            for f in file_list:
                obj_idx = int(f.split('_')[-1].split('.')[0])//sample
                obj = self.obj_list[obj_idx]
                with open(osp.join(self.result_path, f), 'r') as file:
                    data = json.load(file)
                    if data['flag'] == True:
                        if np.array(data['contact_dist']).min() < -0.03:
                            SR[(robot, obj)].append(1)
                        # if np.unique(np.array(data['contact_link'])).shape[0] > :
                        #     SR[(robot, obj)].append(0)
                        else:
                            SR[(robot, obj)].append(1)
                    elif data['flag'] == "Error":
                        pass
                    else:
                        SR[(robot, obj)].append(0)
        self.SR = SR
        # self.result_path = osp.join(osp.dirname(osp.abspath(__file__)), 'results_SR')
        for robot in ['Shadow', 'Allegro', 'Barrett', 'Robotiq']:
            for obj in self.obj_list.values():
                print(robot, obj, np.mean(self.SR[(robot, obj)]))   

    def load_SR_mu(self):
        def extract_middle_integer_sim(filename):
            match = re.search(r'_0.([0-9]+)\_', filename)
            return int(match.group(1)) if match else float('inf')
        SR = {}
        for robot in ['Shadow', 'Allegro', 'Barrett', 'Robotiq']:
            for mu in [0.1, 0.3, 0.5, 0.7, 0.9]:
                SR[(robot, mu)] = []
    

        for robot in ['Shadow', 'Allegro', 'Barrett', 'Robotiq']:
            unsort_file_list = [f for f in os.listdir(self.result_path) if robot in f]
            file_list = sorted(unsort_file_list, key=extract_middle_integer_sim)
            for f in file_list:
                mu = float(f.split('_')[-2])
                with open(osp.join(self.result_path, f), 'r') as file:
                    data = json.load(file)
                    if data['flag'] == True:
                        if np.array(data['contact_dist']).min() < -0.01:
                            SR[(robot, mu)].append(1)
                        # if np.unique(np.array(data['contact_link'])).shape[0] > 6:
                        #     SR[(robot, mu)].append(0)
                        else:
                            SR[(robot, mu)].append(1)
                    elif data['flag'] == "Error":
                        pass
                    else:
                        SR[(robot, mu)].append(0)
        self.SR = SR
        total = []
        for robot in ['Shadow', 'Allegro', 'Barrett', 'Robotiq']:
            for mu in [0.1, 0.3, 0.5, 0.7, 0.9]:
                for instance in SR[(robot, mu)]:
                    total.append(instance)
                print(robot, mu, np.mean(self.SR[(robot, mu)]))
        
        print("total success",np.mean(total)) # 0.481 0.344

    def load_GF_obj(self):
        GF = {}
        for robot in ['Shadow', 'Allegro', 'Barrett', 'Robotiq']:
            for obj in self.obj_list.values():
                GF[(robot, obj)] = []
        for robot in ['Shadow', 'Allegro', 'Barrett', 'Robotiq']:
            if robot == 'Shadow':
                sample = 5
            else:
                sample = 20
            for mu in [0.1, 0.3, 0.5, 0.7, 0.9]:
                file_list = os.listdir(osp.join(self.result_path, robot, 'mu'+str(mu)))
                for f in file_list:
                    obj_idx = int(f.split('_')[0])//sample
                    obj = self.obj_list[obj_idx]
                    with open(osp.join(self.result_path, robot, 'mu'+str(mu), f), 'r') as file:
                        data = json.load(file)
                        try:
                            GF[(robot, obj)].append(data['gf_0'])
                        except:
                            pass
        self.GF = GF    
        #print(self.GF)
        for robot in ['Shadow', 'Allegro', 'Barrett', 'Robotiq']:
            for obj in self.obj_list.values():
                print(robot, obj, np.mean(self.GF[(robot, obj)]))
    def load_GF_mu(self):
        pass

    def load_fctime_robot(self):
        fct = {}
        for robot in ['Shadow', 'Allegro', 'Barrett', 'Robotiq']:
            fct[robot] = []
            for mu in [0.9]:
                file_list = os.listdir(osp.join(self.result_path, robot, 'mu'+str(mu)))
                for f in file_list:
                    with open(osp.join(self.result_path, robot, 'mu'+str(mu), f), 'r') as file:
                        data = json.load(file)
                        try:
                            fct[robot].append(data['fc_time'])
                        except:
                            pass
        self.fct = fct
        print(self.fct)
        # for robot in ['Shadow', 'Allegro', 'Barrett', 'Robotiq']:
        #     print(robot, self.fct[robot])
    
    def load_qtime_robot(self):
        fct = {}
        for robot in ['Shadow', 'Allegro', 'Barrett', 'Robotiq']:
            fct[robot] = []
            for mu in [ 0.9]:
                file_list = os.listdir(osp.join(self.result_path, robot, 'mu'+str(mu)))
                for f in file_list:
                    with open(osp.join(self.result_path, robot, 'mu'+str(mu), f), 'r') as file:
                        data = json.load(file)
                        try:
                            fct[robot].append(data['q_time'])
                            print(data['q_time'])
                        except:
                            traceback.print_exc()
                            pass
        self.qt = fct
        print(self.qt)
        # for robot in ['Shadow', 'Allegro', 'Barrett', 'Robotiq']:
        #     print(robot, self.fct[robot])

    def plot_SR_obj(self):
        self.load_SR_obj()
        robot_list = {}
        for robot in ['Shadow', 'Allegro', 'Barrett', 'Robotiq']:
            robot_list[robot] = []
            for obj in self.obj_list.values():
                robot_list[robot].append(np.mean(self.SR[(robot, obj)]))
        labels = list(self.obj_list.values())
        x = np.arange(len(labels))
        width = 0.15
        fig, ax = plt.subplots(figsize=(14, 5),dpi=300)
        colors = ['#05668D', '#028090', '#00A896', '#02C39A']
        for i, robot in enumerate(['Shadow', 'Allegro', 'Barrett', 'Robotiq']):
            ax.bar(x + (i-1.5)*width, robot_list[robot], width, label=robot, color=colors[i])
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate over Objects')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=80)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.margins(x=0.05) 
        ax.legend(['Shadow', 'Allegro', 'Barrett', 'Robotiq'],loc=3,fontsize=8)
        fig.tight_layout()
        plt.savefig("SR_obj_{}.png".format(self.task),dpi=300,facecolor='#FFFFFF', edgecolor='none')

        plt.show()
        #print(robot_list)

    def plot_SR_mu(self):
        self.load_SR_mu()
        robot_list = {}
        for robot in ['Shadow', 'Allegro', 'Barrett', 'Robotiq']:
            robot_list[robot] = []
            for obj in [0.1, 0.3, 0.5, 0.7, 0.9]:
                robot_list[robot].append(np.mean(self.SR[(robot, obj)]))
        print(robot_list)
        labels = list(['0.1', '0.3', '0.5', '0.7', '0.9'])
        x = np.arange(len(labels))
        width = 0.2
        fig, ax = plt.subplots(figsize=(8, 4),dpi=300)
        for i, robot in enumerate(['Shadow', 'Allegro', 'Barrett', 'Robotiq']):
            ax.bar(x + i*width, robot_list[robot], width, label=robot)
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate over Objects')
        ax.set_xticks(x)
        # ax.set_xticklabels(labels, rotation=80)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(['Shadow', 'Allegro', 'Barrett', 'Robotiq'],loc=0)
        fig.tight_layout()
        plt.savefig("SR_mu_{}.png".format(self.task),dpi=300,facecolor='#FFFFFF', edgecolor='none')

        plt.show()

    # def plot_fc_time(self):
    #     self.load_fctime_robot()
    #     self.load_qtime_robot()
    #     robot_list = {}
    #     for robot in ['Shadow', 'Allegro', 'Barrett', 'Robotiq']:
    #         robot_list[robot] = []
    #         robot_list[robot].append(np.mean(self.fct[robot]))
    #         robot_list[robot].append(np.mean(self.qt[robot]))

    #     labels = ['fc_time', 'q_time']
    #     x = np.arange(len(labels))
    #     width = 0.2
    #     fig, ax = plt.subplots(figsize=(8, 4),dpi=300)
    #     for i, robot in enumerate(['Shadow', 'Allegro', 'Barrett', 'Robotiq']):
    #         ax.bar(x + i*width, robot_list[robot], width, label=robot)
    #     ax.set_ylabel('Computing Time')
    #     ax.set_title('Computing Time over Robots')
    #     ax.set_xticks(x)
    #     # ax.set_xticklabels(labels, rotation=80)
    #     ax.set_xticks(x)
    #     ax.set_xticklabels(labels)
    #     ax.legend(['Shadow', 'Allegro', 'Barrett', 'Robotiq'],loc=1)
    #     fig.tight_layout()
    #     plt.savefig("fc_time.png",dpi=300,facecolor='#FFFFFF', edgecolor='none')
    #     plt.show()
    def plot_only_qtime(self):
        self.load_qtime_robot()
        robot_list = {}
        robots = ['Shadow', 'Allegro', 'Barrett', 'Robotiq']
        robot_data = {robot: [np.mean(self.qt[robot])] for robot in robots}
        error_data = {robot: [stats.sem(self.qt[robot])] for robot in robots}
        print(robot_data)
        x = np.arange(len(robots))  # X positions for robots
        width = 0.3  # Bar width
        fig, ax = plt.subplots(figsize=(7, 7/2), dpi=300)
        colors = ['#E9D8A6']
        # Plot bars for each time category
        for i, label in enumerate(['q_time']):
            times = [robot_data[robot][i] for robot in robots]
            yerr = [error_data[robot][i] for robot in robots]
            ax.bar(x + i * width, times, width, label=label, yerr=yerr, capsize=5, color=colors[i])
            # ax.bar(x + i * width, times, width, label=label)
        # Update labels and title
        ax.set_ylabel('Computing Time')
        ax.set_title('Computing Time over Robots')
        ax.set_xticks(x + width / 2)  # Center the tick labels
        ax.set_xticklabels(robots)  # Set x-axis labels to robot names
        ax.legend(title="Time Type")  # Set the legend for time types
        fig.tight_layout()
        plt.ylim(0, max([np.mean(self.qt[robot]) for robot in robots]) + max([stats.sem(self.qt[robot]) for robot in robots]) + 2)
        plt.savefig("q_time.png", dpi=300, facecolor='#FFFFFF', edgecolor='none')
        plt.show()
        # print(robot_list)

    def plot_fc_time(self):
        self.load_fctime_robot()
        self.load_qtime_robot()

        # Data restructuring: Use robots as x-labels, time categories as bars
        robots = ['Shadow', 'Allegro', 'Barrett', 'Robotiq']
        labels = ['fc_time', 'kine_time']  # Becomes the legend now

        robot_data = {robot: [np.mean(self.fct[robot]), np.mean(self.qt[robot])] for robot in robots}
        error_data = {robot: [stats.sem(self.fct[robot]), stats.sem(self.qt[robot])] for robot in robots}
        print(robot_data)
        x = np.arange(len(robots))  # X positions for robots
        width = 0.3  # Bar width

        fig, ax = plt.subplots(figsize=(7, 7/2), dpi=300)
        colors = ['#E9D8A6', '#005F73']
        # Plot bars for each time category
        for i, label in enumerate(labels):
            times = [robot_data[robot][i] for robot in robots]  # Extract fc_time/q_time across all robots
            yerr = [error_data[robot][i] for robot in robots]  # Extract error bars across all robots
            ax.bar(x + i * width, times, width, label=label, yerr=yerr, capsize=5, color=colors[i])
            # ax.bar(x + i * width, times, width, label=label)

        # Update labels and title
        ax.set_ylabel('Computing Time')
        ax.set_title('Computing Time over Robots')
        ax.set_xticks(x + width / 2)  # Center the tick labels
        ax.set_xticklabels(robots)  # Set x-axis labels to robot names
        ax.legend(title="Time Type")  # Set the legend for time types

        fig.tight_layout()
        plt.ylim(0, max([np.mean(self.qt[robot]) for robot in robots]) + max([stats.sem(self.qt[robot]) for robot in robots]) + 2)
        plt.savefig("fc_time.png", dpi=300, facecolor='#FFFFFF', edgecolor='none')
        plt.show()
    
class sim_result_loader(Dataset):
    def __init__(self,robot,mu,repeat=1):
        super(sim_result_loader, self).__init__()
        self.robot = robot
        self.mu = mu
        self.repeat = repeat
        self.result_path = osp.join(osp.dirname(osp.abspath(__file__)),'CtcSim','bullet', 'sim_result')
        self.file_list = [f for f in os.listdir(self.result_path) if robot in f and (str(mu) in f)]
        print(len(self.file_list))
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        file_name = self.robot + '_' + str(self.mu) + '_' + str(idx) + '.json'
        with open(osp.join(self.result_path, file_name), 'r') as file:
            data = json.load(file)
        if data['flag'] == True:
            return 1
        elif data['flag'] == False:
            return 0
        else:
            print(data['flag'])
            return -1

class similarity_loader(Dataset):
    def __init__(self, robot, mu, device='cuda:1', exp_name='genhand'):
        dexycb_dir = '/home/liyuan/DexYCB/'
        os.environ["DEX_YCB_DIR"] = dexycb_dir
        self.getdata = DexYCBDataset('s0', "test")

        data_dir = osp.join(osp.dirname(osp.abspath(__file__)), "CtcSDF", "data")
        if robot == "Shadow":
            data_sample = 20
        else:
            data_sample = 20

        self.robot_name = robot
        self.mu = mu
        self.device = device
        self.exp = exp_name
        if exp_name == "genhand":
            self.result_dir = osp.join(osp.abspath(osp.dirname(__file__)),  "results", self.robot_name, "mu"+str(mu))
        elif exp_name == "nv":
            self.result_dir = osp.join(osp.abspath(osp.dirname(__file__)),  "results_nv", self.robot_name, "mu"+str(mu))

        config_file = osp.join(data_dir, "dexycb_test_s0.json")
        with open(config_file, 'r') as f:
            self.config = json.load(f)

        sample_dir = osp.join(data_dir, "test_sample_{}.json".format(data_sample))
        with open(sample_dir, 'r') as f:
            sample = json.load(f)
            all_indices = [np.array(i) for i in sample.values()]
            self.sample = np.concatenate(all_indices)     
        self.len = len(self.sample)   
        self.mano_layer = ManoLayer(
                    ncomps = 45,
                    side = 'right',
                    mano_root = os.path.join(os.path.dirname(__file__), 'manopth','mano','models'),
                    use_pca=True,
                    flat_hand_mean=False
                    )
        self.faces = torch.LongTensor(np.load(osp.join(osp.dirname(osp.abspath(__file__)), 'CtcSDF','closed_fmano.npy')))            
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx_):
        idx = self.sample[idx_]
        s0_id = self.config['images'][idx]['id']
        sample = self.getdata[s0_id]
        ycb_id = sample['ycb_grasp_ind']
        label = np.load(sample['label_file'])
        pose_y = label['pose_y'][ycb_id]

        obj_file = self.getdata.obj_file[sample['ycb_ids'][ycb_id]]
        # print("dir check", obj_file)
        pose = np.vstack((pose_y, np.array([[0, 0, 0, 1]], dtype=np.float32)))
        hand_trans = np.array(self.config['annotations'][idx]['hand_trans'], dtype = np.float32)
        # translation = pose[:3, 3]- hand_trans
        pose[:3, 3] = pose[:3, 3]- hand_trans
        
        pose_m = torch.from_numpy(label['pose_m'])
        bates = torch.tensor(sample['mano_betas'], dtype=torch.float32).unsqueeze(0)
        verts,_,_,_,_,_ = self.mano_layer( pose_m[:, 0:48],bates, pose_m[:, 48:51])
        # print("hand trans", pose_m[:, 48:51])
        verts = verts.view(778,3).numpy()
        # pose_y_ = np.vstack((label['pose_y'][ycb_id],np.array([[0, 0, 0, 1]], dtype=np.float32)))

        obj_centre = np.array(self.config['annotations'][idx]['obj_center_3d'])
        obj_centre = obj_centre - hand_trans
        
        obj_mesh = tm.load_mesh(obj_file, force='mesh', process=False)
        pose[:3, 3] = pose[:3, 3] - obj_centre 
        obj_mesh.apply_transform(pose)
        points, face_ids = tm.sample.sample_surface_even(obj_mesh, 1024)
        obj_verts = obj_mesh.vertices
        obj_faces = obj_mesh.faces
        obj_points = points
        obj_normal = obj_mesh.face_normals[face_ids]
        
        result_file = str(idx_) + "_" + self.robot_name + "_" + str(0) + "_mu_" + str(self.mu) + ".json"
        try:
            q, w, x = self.opt_result_parser(osp.join(self.result_dir, result_file))
        except:
            traceback.print_exc()
            q = 0
            w = 0
            x = 0
        return {'hand_vert': torch.tensor((verts-obj_centre), dtype=torch.float32), 
                'hand_face': torch.tensor(self.faces, dtype=torch.float32), 
                'q': torch.tensor(q, dtype=torch.float32), 
                'obj_vert': torch.tensor(obj_verts, dtype=torch.float32), 
                'obj_face': torch.tensor(obj_faces, dtype=torch.float32),
                'obj_points': torch.tensor(obj_points, dtype=torch.float32),
                'obj_normal': torch.tensor(obj_normal, dtype=torch.float32),
                'w': torch.tensor(w, dtype=torch.float32),
                'x': torch.tensor(x, dtype=torch.float32),
                }
    
    def opt_result_parser(self, file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
        if self.exp == "genhand":
            q = np.array(data['qr']).squeeze()
        if self.exp == "nv":
            q = np.array(data['q']).squeeze()
        if self.robot_name == "Robotiq":
            joint_val = q[6:] * np.array([-1, -1, 1, -1, 1, -1])
            q = np.concatenate([q[:6], joint_val])
        try:
            w = np.array(data['w']).squeeze()     
            x = np.array(data['x']).squeeze() 
            return q, w, x    
        except:

            return q ,0, 0

class similarity:
    def __init__(self, robot, mu, device='cuda:0', exp_name='genhand'):
        self.robot = robot
        self.mu = mu
        self.device = device
        self.exp_name = exp_name
        if robot == 'Shadow':
            rob = Shadow(batch=1, device=device)
            self.sample = 20
        elif robot == 'Allegro':
            rob = Allegro(batch=1, device=device)
            self.sample = 20
        elif robot == 'Barrett':
            rob = Barrett(batch=1, device=device)
            self.sample = 20
        elif robot == 'Robotiq':
            rob = Robotiq(batch=1, device=device)
            self.sample = 20
        self.dataset = similarity_loader(robot, mu, device, exp_name)
        self.rob = rob
    
    def get_contact_map(self, contactpack):
        sdf_val, sdf_normal = self.pv_sdf_hand(contactpack[...,:3])
        points, normals = contactpack[...,:3], contactpack[...,3:]
        condition_sdf = sdf_val <= self.contact_threshold
        normal_similarity = torch.sum(normals * sdf_normal, dim=-1)
        condition_antipodal = normal_similarity < self.cosine_threshold
        point_indices = torch.nonzero(condition_sdf & condition_antipodal, as_tuple=True)[0]
        return points[point_indices,:]
    
    def get_contact_sdf_residual(self):
        """
        for Robotiq
        """
        key_points = torch.stack(self.rob.get_fingertip_surface_points_updated())
        key_points = key_points.view(-1,3)
        num_points = key_points.size(0)

        keyset_1 = key_points[num_points//2:]
        keyset_2 = key_points[:num_points//2]
        sdf_val_1, sdf_normal = self.pv_sdf(keyset_1)
        valid_sdf_1 = sdf_val_1 <= 0.05
        sdf_min_1 = sdf_val_1[valid_sdf_1].min()
        sdf_val_2, sdf_normal = self.pv_sdf(keyset_2)
        valid_sdf_2 = sdf_val_2 <= 0.05
        sdf_min_2 = sdf_val_2[valid_sdf_2].min()

        mean_error = torch.mean(torch.tensor([sdf_min_1, sdf_min_2]))  # corrected here
        # print("mean error", mean_error.item())
        return mean_error.item()
    
    def get_contact_sdf_residual_(self):
        key_points = self.rob.get_keypoints()
        key_points = key_points.view(-1,3)[1:]
        print("key_points", key_points)
        sdf_val, sdf_normal = self.pv_sdf(key_points)
        print("sdf_val", sdf_val)
        valid_sdf = sdf_val <= 0.05
        print("valid sdf", valid_sdf)
        mean_error = torch.mean(sdf_val[valid_sdf])
        # print("mean error", mean_error.item())
        return mean_error.item()

    def get_gf_residual_nv(self):
        key_points = self.rob.get_keypoints()
        key_points = key_points.view(-1,3)[1:]
        sdf_val, sdf_normal = self.pv_sdf(key_points)
        valid_sdf = sdf_val <= 0.05
        x = key_points[valid_sdf]
        len_x = x.size(0)

        w = torch.ones((len_x,4),dtype=torch.float32).to(self.device)
        w *= 0.25
        gf = self.obj.gf_residual(x.view(1,-1,3), w.view(1,-1,4))
        print("gf", gf)
        return gf["Gf"]
    
    def get_gf_residual(self):
        assert self.w is not None and self.x is not None and self.x.size(0) == self.w.size(0)
        key_points = self.rob.get_keypoints(subset=self.x.size(0))

        cdist = torch.cdist(key_points.view(-1,3), self.x.view(-1,3))
        row_id, col_id = linear_sum_assignment(cdist.cpu().numpy())

        w = self.w.view(-1,4)[col_id]
        gf = self.obj.gf_residual(key_points.view(1,-1,3), w.view(1,-1,4))
        print("gf",gf["Gf"])


        return gf["Gf"]

    def get_dist_residual(self):
        assert self.w is not None and self.x is not None and self.x.size(0) == self.w.size(0)
        key_points = self.rob.get_keypoints(subset=self.x.size(0))
        cdist = torch.cdist(key_points.view(-1,3), self.x.view(-1,3))
        row_ind, col_ind = linear_sum_assignment(cdist.cpu().numpy())
        distance_loss = (torch.sum(cdist[row_ind, col_ind])/self.x.size(0)) #+ distance_0
        print("distance loss", distance_loss.item())
        return distance_loss.item()

    def get_contact_map_robot(self, contactpack):
        slice_id = []
        for sdf in self.rob_sdf:
            
            sdf_val, sdf_normal = sdf(contactpack[...,:3])
            points, normals = contactpack[...,:3], contactpack[...,3:]
            condition_sdf = sdf_val <= self.contact_threshold
            normal_similarity = torch.sum(normals * sdf_normal, dim=-1)
            condition_antipodal = normal_similarity < self.cosine_threshold
            point_indices = torch.nonzero(condition_sdf & condition_antipodal, as_tuple=True)[0]
            slice_id.append(point_indices)
        slice_id = torch.cat(slice_id, dim=0)
        uni_id = torch.unique(slice_id)
        return points[uni_id,:]

        #points_set.append(points[:,point_indices,:])
        # return torch.cat(points_set, dim=1)
    
    def initialize(self, datapack):
        hand_vert = datapack['hand_vert'].to(self.device)
        hand_face = datapack['hand_face'].to(self.device)
        obj_points = datapack['obj_points'].to(self.device)
        obj_normal = datapack['obj_normal'].to(self.device)
        obj_face = datapack['obj_face'].to(self.device)
        obj_verts = datapack['obj_vert'].to(self.device)
        q = datapack['q'].to(self.device)
        self.pv_sdf_hand = pv.MeshSDF(
            pv.MeshObjectFactory(
                mesh_name=None, 
                preload_mesh=dict(vertices=hand_vert.squeeze().cpu().numpy(), 
                                  faces=hand_face.squeeze().cpu().numpy()))) #the backend open3d is not compatible with cuda
        
        self.pv_sdf = pv.MeshSDF(
            pv.MeshObjectFactory(
                mesh_name=None,
                preload_mesh=dict(vertices=obj_verts.squeeze().cpu().numpy(), 
                                  faces=obj_face.squeeze().cpu().numpy())))
        
        self.rob.forward_kinematics(q)
        # self.draw(obj_verts.squeeze().cpu().numpy(), obj_face.squeeze().cpu().numpy())
        self.obj = object_sdf(device=self.device, contact_threshold=0.020)
        self.obj.reset([obj_face, obj_verts], [hand_vert, hand_face], None)
        self.w = datapack['w'].to(self.device)
        self.x = datapack['x'].to(self.device)
        #build the sdf for the robot
        self.rob_sdf = []
        for idx, link in enumerate(self.rob.meshV):
            V = self.rob.meshV[link]
            F = self.rob.meshF[link]
            self.rob_sdf.append(pv.MeshSDF(
                pv.MeshObjectFactory(
                    mesh_name=None,
                    preload_mesh=dict(vertices=V, faces=F))))


        self.obj_verts = obj_points
        self.obj_normals = obj_normal

        self.contact_threshold = 0.02
        self.cosine_threshold = 0

    def upsample_points(self, points, target_size):
        """ Upsamples a set of points to a target size by repeating points randomly. """
        repeat_factor = target_size // points.size(0) + 1
        repeated_points = points.repeat(repeat_factor, 1)
        
        # Now randomly select 'target_size' points from the repeated points
        indices = torch.randperm(repeated_points.size(0))[:target_size]
        upsampled_points = repeated_points[indices]
        
        return upsampled_points
    def downsample_points(self, points, target_size):
        """ Randomly downsamples a set of points to a target size. """
        # Generate a random permutation of indices
        indices = torch.randperm(points.size(0))
        
        # Select the first 'target_size' indices
        downsampled_indices = indices[:target_size]
        
        # Use these indices to select points
        downsampled_points = points[downsampled_indices]
        
        return downsampled_points
    def chamfer_distance__(self, set1, set2):
        """
        Calculate the Chamfer Distance between two point sets
        :param set1: tensor of shape (N, D) where N is points and D is dimension
        :param set2: tensor of shape (M, D) where M is points and D is dimension
        :return: Chamfer Distance between the two point sets
        """
        n_points = set1.size(0)
        m_points = set2.size(0)
        
        # Expand set1 to match the points in set2
        set1_expanded = set1.unsqueeze(1).expand(n_points, m_points, -1)
        set2_expanded = set2.unsqueeze(0).expand(n_points, m_points, -1)
        
        # Compute squared distances between all points (N, M)
        distances = torch.norm(set1_expanded - set2_expanded, dim=2, p=2)
        
        # Get min distance from each point in set1 to points in set2 (N,)
        min_distances_set1_to_set2 = torch.min(distances, dim=1)[0]
        
        # Get min distance from each point in set2 to points in set1 (M,)
        min_distances_set2_to_set1 = torch.min(distances, dim=0)[0]
        
        # Average the distances
        chamfer_dist = torch.mean(min_distances_set1_to_set2) + torch.mean(min_distances_set2_to_set1)
        
        return chamfer_dist
    
    def KD_chamfer_distance(self, set1, set2):
        set1 *= 100
        set2 *= 100
        set1 = set1.cpu().numpy()
        set2 = set2.cpu().numpy()
        tree1 = KDTree(set1)
        direct_1, _ = tree1.query(set2)
        # set1_to_set2 = np.mean(np.square(direct_1))
        set1_to_set2 = np.mean(direct_1)
        tree2 = KDTree(set2)
        direct_2, _ = tree2.query(set1)
        # set2_to_set1 = np.mean(np.square(direct_2))
        set2_to_set1 = np.mean(direct_2)

        return set1_to_set2 + set2_to_set1


    
    def run_similarity(self):
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=4)
        chamfer = []
        bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Computing Chamfer Distance")
        for idx, datapack in bar:
            try:
                self.initialize(datapack)
                hand_contact = self.get_contact_map(torch.cat([self.obj_verts.view(-1,3), self.obj_normals.view(-1,3)], dim=-1))
                rob_contact = self.get_contact_map_robot(torch.cat([self.obj_verts.view(-1,3), self.obj_normals.view(-1,3)], dim=-1))
                # print("Hand Contact Points: ", hand_contact.size())
                if hand_contact.size(0) == 0 or rob_contact.size(0) == 0:
                    continue
                # c_dist = chamfer_distance(hand_contact.unsqueeze(0), rob_contact.unsqueeze(0),single_directional=False)
                c_dist = self.KD_chamfer_distance(hand_contact, rob_contact)

                chamfer.append(c_dist)
            except:
                traceback.print_exc()
                continue
        return torch.tensor(chamfer)

    def run_sdf_residual(self):
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=4)
        chamfer = []
        bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Computing sdf Residual")
        for idx, datapack in bar:
            try:
                self.initialize(datapack)
                if self.robot == 'Robotiq':
                    sdf_residual = self.get_contact_sdf_residual()
                else:
                    sdf_residual = self.get_contact_sdf_residual_()
                chamfer.append(sdf_residual)
            except:
                traceback.print_exc()
                continue
        return torch.tensor(chamfer)


    
    def run_dist_residual(self):
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=4)
        chamfer = []
        bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Computing sdf Residual")
        for idx, datapack in bar:
            try:
                self.initialize(datapack)
                sdf_residual = self.get_dist_residual()
                chamfer.append(sdf_residual)
            except:
                traceback.print_exc()
                continue
        return torch.tensor(chamfer)
    
    def run_gf_residual(self):
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=4)
        chamfer = []
        bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Computing sdf Residual")
        for idx, datapack in bar:
            try:
                self.initialize(datapack)
                sdf_residual = self.get_gf_residual()
                chamfer.append(sdf_residual)
            except:
                traceback.print_exc()
                continue
        return torch.tensor(chamfer)
    def run_gf_residual_nv(self):
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=4)
        chamfer = []
        bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Computing sdf Residual")
        for idx, datapack in bar:
            try:
                self.initialize(datapack)
                sdf_residual = self.get_gf_residual_nv()
                chamfer.append(sdf_residual)
            except:
                traceback.print_exc()
                continue
        return torch.tensor(chamfer)
    def run_sdf_residual_test(self):
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=4)
        chamfer = []
        datapack = next(iter(dataloader))
        try:
            self.initialize(datapack)
            sdf_residual = self.get_contact_sdf_residual()
            chamfer.append(sdf_residual)
        except:
            traceback.print_exc()
        return torch.tensor(chamfer)

    
    def similarity_idx(self, idx):
        datapack = self.dataset[idx]
        datapack['q'] = datapack['q'].unsqueeze(0)
        self.initialize(datapack)

        hand_contact = self.get_contact_map(torch.cat([self.obj_verts, self.obj_normals], dim=-1))
        rob_contact = self.get_contact_map_robot(torch.cat([self.obj_verts, self.obj_normals], dim=-1))
        print("Hand Contact Points: ", hand_contact.size())
        print("Robot Contact Points: ", rob_contact.size())
        c_dist = self.KD_chamfer_distance(hand_contact, rob_contact)        
        print("Chamfer Distance: ", c_dist)
        return hand_contact, rob_contact
    
    def draw(self, verts, faces):
        from plotly import graph_objects as go
        data = self.rob.get_mesh_updated()
        data.append(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                                        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                        opacity=0.5))
        fig = go.Figure(data=data)
        fig.update_layout(scene=dict(
            xaxis=dict(nticks=4, range=[-1, 1]),
            yaxis=dict(nticks=4, range=[-1, 1]),
            zaxis=dict(nticks=4, range=[-1, 1])
        ))
        fig.show()

    
def test_similarity():
    idx = 0
    sim = similarity('Allegro', 0.9, exp_name='nv')
    h,r = sim.similarity_idx(idx)
    sim = similarity('Allegro', 0.9)
    h,r = sim.similarity_idx(idx)
        
def test_dataset():
    loader = similarity_loader('Shadow', 0.9)
    datapack = loader[0]
    hand_vert = datapack['hand_vert']
    hand_face = datapack['hand_face']
    obj_vert = datapack['obj_vert']
    obj_face = datapack['obj_face']
    q = datapack['q']
    sim = similarity('Robotiq', 0.9)
    c = sim.similarity_idx(315)
    # data_viz("meshjoint", (obj_vert, obj_face, None, h.squeeze().cpu().numpy()))
    # data_viz("meshjoint", (obj_vert, obj_face, None, r.squeeze().cpu().numpy()))
    # data_viz("meshmesh", (hand_vert, hand_face, obj_vert, obj_face, None))
    # data_viz("mesh", ())

def test_opt():
    result = result_loader('gh', 'sim', 'gt')
    # # result.load_qtime_robot()
    # # result.load_fctime_robot()
    # # result.plot_fc_time()
    # result.load_SR_mu()
    # result.plot_SR_mu()
    # result = result_loader('gh', 'opt','gt')
    # result.load_qtime_robot()
    # result.load_fctime_robot()
    # result.plot_fc_time()
    # result.load_SR_mu()
    result.plot_SR_mu()
    # result.plot_only_qtime()
    # result.plot_SR_obj()
def main_():
    chamfer_score_nv = {}
    for robot in ['Robotiq', 'Shadow', 'Allegro', 'Barrett']:
        for mu in [0.9]:
            sim = similarity(robot, mu, device='cpu', exp_name='genhand')
            score = sim.run_dist_residual()
            residual_tensor = torch.tensor(score)
            non_nan_tensor = residual_tensor[~torch.isnan(residual_tensor)]
            chamfer_score_nv[(robot)] = non_nan_tensor.numpy()
    robot_list_method1 = {}
    error_margin_method1 = {}
    
    for robot in ['Shadow', 'Allegro', 'Barrett', 'Robotiq']:
        robot_list_method1[robot] = []
        error_margin_method1[robot] = []

        for obj in [0.9]:
            # Method 1
            mean_val_method1 = np.mean(chamfer_score_nv[(robot)])*100
            std_val_method1 = np.std(chamfer_score_nv[(robot)])*100
            robot_list_method1[robot].append(mean_val_method1)
            error_margin_method1[robot].append(std_val_method1)
    
    robots = ['Shadow', 'Allegro', 'Barrett', 'Robotiq']
    labels = ['5-finger', '4-finger', '3-finger', '2-finger']
    x = np.arange(len(robots))
    
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=300)
    method1_mean = [robot_list_method1[robot][0] for robot in robots]
    method1_error = [error_margin_method1[robot][0] for robot in robots]
    ax.plot(x, method1_mean, label="Ours", color='#005F73', linewidth=2)
    ax.fill_between(x, np.array(method1_mean) - np.array(method1_error), np.array(method1_mean) + np.array(method1_error), color='#005F73', alpha=0.2, edgecolor=None)
    ax.set_ylabel('Distance Residual (cm)', color='#005F73')
    ax.set_title('Kinematics optimisation distance residual (mu = 0.9)')
    ax.set_xticklabels(labels)  # Use robot names as x-axis labels
    ax.set_xticks(x)  # Set the x-ticks at the robot positions

    ax.tick_params(axis='y', labelcolor='#005F73')
    # ax.set_xticks(x)
    ax.legend(loc='best')
    fig.tight_layout()
    plt.savefig("SR_mu_comparison_0.9_Distance_residual.png", dpi=300, facecolor='#FFFFFF', edgecolor='none')
    plt.show()
    
def save_dict_to_json(dict_of_arrays, file_path):
    # Convert NumPy arrays to lists
    dict_of_lists = {f"{key}": value.tolist() for key, value in dict_of_arrays.items()}
    
    # Save the dictionary to a JSON file
    with open(file_path, 'w') as f:
        json.dump(dict_of_lists, f)


def evaluate():
    # chamfer_score_nv = {}
    # for robot in ['Robotiq', 'Shadow', 'Allegro', 'Barrett']:
    #     for mu in [0.1,0.3,0.5,0.7,0.9]:
    #         sim = similarity(robot, mu, device='cpu', exp_name='genhand')
    #         score = sim.run_gf_residual()
    #         residual_tensor = torch.tensor(score)
    #         non_nan_tensor = residual_tensor[~torch.isnan(residual_tensor)]
    #         chamfer_score_nv[(robot, mu)] = non_nan_tensor.numpy()
    
    # save_dict_to_json(chamfer_score_nv, 'gf_residual_gh.json')
    
    # chamfer_score_nv_2 = {}
    # for robot in ['Robotiq', 'Shadow', 'Allegro', 'Barrett']:
    #     for mu in [0.1,0.3,0.5,0.7,0.9]:
    #         sim = similarity(robot, mu, device='cpu', exp_name='nv')
    #         score = sim.run_gf_residual_nv()
    #         residual_tensor = torch.tensor(score)
    #         non_nan_tensor = residual_tensor[~torch.isnan(residual_tensor)]
    #         chamfer_score_nv_2[(robot, mu)] = non_nan_tensor.numpy()
    
    # save_dict_to_json(chamfer_score_nv_2, 'gf_residual_nv.json')
    
    # chamfer_score_nv_3 = {}
    # for robot in ['Robotiq', 'Shadow', 'Allegro', 'Barrett']:
    #     for mu in [0.1,0.3,0.5,0.7,0.9]:
    #         sim = similarity(robot, mu, device='cpu', exp_name='genhand')
    #         score = sim.run_sdf_residual()
    #         residual_tensor = torch.tensor(score)
    #         non_nan_tensor = residual_tensor[~torch.isnan(residual_tensor)]
    #         chamfer_score_nv_3[(robot, mu)] = non_nan_tensor.numpy()
    
    # save_dict_to_json(chamfer_score_nv_3, 'sdf_residual_gh.json')
    
    # chamfer_score_nv_4 = {}
    # for robot in ['Robotiq', 'Shadow', 'Allegro', 'Barrett']:
    #     for mu in [0.1,0.3,0.5,0.7,0.9]:
    #         sim = similarity(robot, mu, device='cpu', exp_name='nv')
    #         score = sim.run_sdf_residual()
    #         residual_tensor = torch.tensor(score)
    #         non_nan_tensor = residual_tensor[~torch.isnan(residual_tensor)]
    #         chamfer_score_nv_4[(robot, mu)] = non_nan_tensor.numpy()
    
    # save_dict_to_json(chamfer_score_nv_4, 'sdf_residual_nv.json')
    
    
    chamfer_score_nv_5 = {}
    for robot in ['Robotiq', 'Shadow', 'Allegro', 'Barrett']:
        for mu in [0.1,0.3,0.5,0.7,0.9]:
            sim = similarity(robot, mu, device='cpu', exp_name='genhand')
            score = sim.run_dist_residual()
            residual_tensor = torch.tensor(score)
            non_nan_tensor = residual_tensor[~torch.isnan(residual_tensor)]
            chamfer_score_nv_5[(robot, mu)] = non_nan_tensor.numpy()
            print("chamfer_score_nv_5", chamfer_score_nv_5)
    save_dict_to_json(chamfer_score_nv_5, 'distance_residual_gh.json')
    
def draw_gf_residual():
    gh_record = 'gf_residual_gh.json'
    nv_record = 'gf_residual_nv.json'
    with open(osp.join(osp.dirname(__file__),gh_record), 'r') as f:
        gh_data = json.load(f)
    with open(osp.join(osp.dirname(__file__),nv_record), 'r') as f:
        nv_data = json.load(f)
    
    robot_list_gh = {}
    robot_list_nv = {}
    error_margin_gh = {}
    error_margin_nv = {}
    for mu in [0.1,0.3,0.5,0.7,0.9]:
        robot_list_gh[mu] = []
        robot_list_nv[mu] = []
        error_margin_gh[mu] = []
        error_margin_nv[mu] = []
        for robot in ['Robotiq', 'Barrett', 'Allegro', 'Shadow']:
            mean_val_gh = np.mean(gh_data[f"{robot, mu}"])
            std_val_gh = stats.sem(gh_data[f"{robot, mu}"])
            robot_list_gh[mu].append(mean_val_gh)
            error_margin_gh[mu].append(std_val_gh)
            
            mean_val_nv = np.mean(nv_data[f"{robot, mu}"])
            std_val_nv = stats.sem(nv_data[f"{robot, mu}"])
            # std_val_nv = np.std(nv_data[f"{robot, mu}"])
            robot_list_nv[mu].append(mean_val_nv)
            error_margin_nv[mu].append(std_val_nv)
    robots = ['Robotiq', 'Barrett', 'Allegro', 'Shadow']
    mus = [0.1,0.3,0.5,0.7,0.9]
    labels =  ['Robotiq\n2-finger', 'Barrett\n3-finger', 'Allegro\n4-finger', 'Shadow\n5-finger']
    x = np.arange(len(robots))
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 7), dpi=300,
                                   gridspec_kw={'height_ratios': [2, 1]})
    fig.subplots_adjust(hspace=0.05)  # Adjust space between axes
    maker = ['o', '^', 's', 'D', 'P']
    color = ['#005F73', '#CA6702']
    

    for j, mu in enumerate(mus):
        mean_val_gh = robot_list_gh[mu]
        error_val_gh = error_margin_gh[mu]
        ax2.plot(x, mean_val_gh,  label=f"Ours ("+r"$\mu=$"+str(mu)+")", color=color[0], marker=maker[j], markersize=7, linewidth=1)
        ax2.fill_between(x, np.array(mean_val_gh) - np.array(error_val_gh), np.array(mean_val_gh) + np.array(error_val_gh), color=color[0], alpha=0.2, edgecolor=None)
    for j, mu in enumerate(mus):
        mean_val_nv = robot_list_nv[mu]
        error_val_nv = error_margin_nv[mu]
        ax1.plot(x, mean_val_nv, label=f"Baseline ("+r"$\mu=$"+str(mu)+")", color=color[1], marker=maker[j], markersize=7, linewidth=1)
        ax1.fill_between(x, np.array(mean_val_nv) - np.array(error_val_nv), np.array(mean_val_nv) + np.array(error_val_nv), color=color[1], alpha=0.2, edgecolor=None)

    ax1.tick_params(axis='y', labelcolor=color[1])
    ax1.set_xticklabels([])  # Remove x-ticklabels from the upper axis
    ax2.tick_params(axis='y', labelcolor=color[0])
    ax1.legend(loc='best', fontsize=8) 
    ax2.legend(loc='best', fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    
    ax1.set_ylim(4, 28)  # Higher values (Baseline)
    ax2.set_ylim(0, 0.6)  # Lower values (Ours)
        
        
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.set_ticks_position('none')  # no ticks on the upper axis

    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    plt.setp(ax2.get_yticklabels()[-1], visible=False)  # Hide the last y-tick label
    plt.setp(ax1.get_yticklabels()[0], visible=False)  # Hide the last y-tick label
    plt.setp(ax2.get_yticklines()[-2], visible=False)  # Hide the last y-tick label
    plt.setp(ax1.get_yticklines()[0], visible=False)  # Hide the last y-tick label
    # Add a global title for the whole figure
    fig.suptitle('Net wrench residual over Robots', fontsize=12)
    fig.text(0.02, 0.5, 'Net Wrench Residual', ha='center', va='center', rotation='vertical', fontsize=12)
    d = .5  # Proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)

    # Slanted lines between ax1 and ax2
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    fig.tight_layout()

    plt.savefig("SR_mu_comparison_Net_wrench_residual.png", dpi=300, facecolor='#FFFFFF', edgecolor='none')

    plt.show()
    
def draw_gf_residual_single():
    gh_record = 'gf_residual_gh.json'
    nv_record = 'gf_residual_nv.json'
    with open(osp.join(osp.dirname(__file__),gh_record), 'r') as f:
        gh_data = json.load(f)
    with open(osp.join(osp.dirname(__file__),nv_record), 'r') as f:
        nv_data = json.load(f)
    
    robot_list_gh = {}
    robot_list_nv = {}
    error_margin_gh = {}
    error_margin_nv = {}
    for mu in [0.1,0.3,0.5,0.7,0.9]:
        robot_list_gh[mu] = []
        robot_list_nv[mu] = []
        error_margin_gh[mu] = []
        error_margin_nv[mu] = []
        for robot in ['Shadow', 'Allegro', 'Barrett', 'Robotiq']:
            mean_val_gh = np.mean(gh_data[f"{robot, mu}"])
            std_val_gh = stats.sem(gh_data[f"{robot, mu}"])
            robot_list_gh[mu].append(mean_val_gh)
            error_margin_gh[mu].append(std_val_gh)
            
            mean_val_nv = np.mean(nv_data[f"{robot, mu}"])
            std_val_nv = stats.sem(nv_data[f"{robot, mu}"])
            # std_val_nv = np.std(nv_data[f"{robot, mu}"])
            robot_list_nv[mu].append(mean_val_nv)
            error_margin_nv[mu].append(std_val_nv)

    print("robot_list_gh", robot_list_gh)
    print("robot_list_nv", robot_list_nv)
    robots = ['Shadow', 'Allegro', 'Barrett', 'Robotiq']
    mus = [0.1,0.3,0.5,0.7,0.9]
    labels = ['5-finger', '4-finger', '3-finger', '2-finger']
    x = np.arange(len(robots))
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=300)

    maker = ['o', '^', 's', 'D', 'P']
    color = ['#005F73', '#CA6702']
    for j, mu in enumerate(mus):
        mean_val_nv = robot_list_nv[mu]
        error_val_nv = error_margin_nv[mu]
        ax.plot(x, mean_val_nv, label=f"Baseline ("+r"$\mu=$"+str(mu)+")", color=color[1], marker=maker[j], markersize=3, linewidth=1)
        ax.fill_between(x, np.array(mean_val_nv) - np.array(error_val_nv), np.array(mean_val_nv) + np.array(error_val_nv), color=color[1], alpha=0.2, edgecolor=None)
   

    for j, mu in enumerate(mus):
        mean_val_gh = robot_list_gh[mu]
        error_val_gh = error_margin_gh[mu]
        ax.plot(x, mean_val_gh, label=f"Ours ("+r"$\mu=$"+str(mu)+")", color=color[0], marker=maker[j], markersize=3, linewidth=1)
        ax.fill_between(x, np.array(mean_val_gh) - np.array(error_val_gh), np.array(mean_val_gh) + np.array(error_val_gh), color=color[0], alpha=0.2, edgecolor=None)

    ax.set_ylabel('Net wrench residual')
    ax.set_title('Net wrench residual over Robots')
    ax.set_xticks(x)  # Set the x-ticks at the robot positions
    ax.set_xticklabels(labels)  # Use robot names as x-axis labels
    ax.legend(loc='best', fontsize=8)


    fig.tight_layout()

    plt.savefig("SR_mu_comparison_Net_wrench_residual_single.png", dpi=300, facecolor='#FFFFFF', edgecolor='none')

    plt.show()

def draw_sdf_residual_single():
    gh_record = 'sdf_residual_gh.json'
    nv_record = 'sdf_residual_nv.json'
    with open(osp.join(osp.dirname(__file__),gh_record), 'r') as f:
        gh_data = json.load(f)
    with open(osp.join(osp.dirname(__file__),nv_record), 'r') as f:
        nv_data = json.load(f)
    
    robot_list_gh = {}
    robot_list_nv = {}
    error_margin_gh = {}
    error_margin_nv = {}
    for mu in [0.1,0.3,0.5,0.7,0.9]:
        robot_list_gh[mu] = []
        robot_list_nv[mu] = []
        error_margin_gh[mu] = []
        error_margin_nv[mu] = []
        for robot in ['Robotiq', 'Barrett', 'Allegro', 'Shadow']:
            mean_val_gh = np.mean(gh_data[f"{robot, mu}"])*100
            std_val_gh = stats.sem(gh_data[f"{robot, mu}"])*100
            robot_list_gh[mu].append(mean_val_gh)
            error_margin_gh[mu].append(std_val_gh)
            
            mean_val_nv = np.mean(nv_data[f"{robot, mu}"])*100
            std_val_nv = stats.sem(nv_data[f"{robot, mu}"])*100
            # std_val_nv = np.std(nv_data[f"{robot, mu}"])
            robot_list_nv[mu].append(mean_val_nv)
            error_margin_nv[mu].append(std_val_nv)
    print("robot_list_gh", robot_list_gh)
    print("robot_list_nv", robot_list_nv)
    robots = ['Robotiq', 'Barrett', 'Allegro', 'Shadow']
    mus = [0.1,0.3,0.5,0.7,0.9]
    labels =  ['Robotiq\n2-finger', 'Barrett\n3-finger', 'Allegro\n4-finger', 'Shadow\n5-finger']
    x = np.arange(len(robots))
    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

    maker = ['o', '^', 's', 'D', 'P']
    color = ['#005F73', '#CA6702']

    for j, mu in enumerate(mus):
        mean_val_nv = robot_list_nv[mu]
        error_val_nv = error_margin_nv[mu]
        ax.plot(x, mean_val_nv, label=f"Baseline ("+r"$\mu=$"+str(mu)+")", color=color[1], marker=maker[j], markersize=7, linewidth=1)
        ax.fill_between(x, np.array(mean_val_nv) - np.array(error_val_nv), np.array(mean_val_nv) + np.array(error_val_nv), color=color[1], alpha=0.2, edgecolor=None)


    for j, mu in enumerate(mus):
        mean_val_gh = robot_list_gh[mu]
        error_val_gh = error_margin_gh[mu]
        ax.plot(x, mean_val_gh, label=f"Ours ("+r"$\mu=$"+str(mu)+")", color=color[0], marker=maker[j], markersize=7, linewidth=1)
        ax.fill_between(x, np.array(mean_val_gh) - np.array(error_val_gh), np.array(mean_val_gh) + np.array(error_val_gh), color=color[0], alpha=0.2, edgecolor=None)

    ax.set_ylabel('SDF value residual (cm)')
    ax.set_title('SDF value residual over Robots')
    ax.set_xticks(x)  # Set the x-ticks at the robot positions
    ax.set_xticklabels(labels)  # Use robot names as x-axis labels
    ax.legend(loc='best')


    fig.tight_layout()

    plt.savefig("SR_mu_comparison_SDF_residual_single.png", dpi=300, facecolor='#FFFFFF', edgecolor='none')

    plt.show()

def main():
    pass
    # chamfer_score = {}
    # for robot in ['Robotiq', 'Shadow', 'Allegro', 'Barrett']:
    #     for mu in [0.3, 0.7, 0.9]:
    #         sim = similarity(robot, mu, device='cuda:0')
    #         chamfer_score[(robot)] = sim.run_similarity()
    # print(chamfer_score)
    

    # robot_list_method1 = {}
    # robot_list_method2 = {}
    # error_margin_method1 = {}
    # error_margin_method2 = {}
            
    # for robot in ['Shadow', 'Allegro', 'Barrett', 'Robotiq']:
    #     robot_list_method1[robot] = []
    #     robot_list_method2[robot] = []
    #     error_margin_method1[robot] = []
    #     error_margin_method2[robot] = []

    #     for obj in [0.9]:  # Only consider mu = 0.9
    #         # Method 1
    #         mean_val_method1 = np.mean(chamfer_score_nv[(robot)])
    #         std_val_method1 = np.std(chamfer_score_nv[(robot)])
    #         robot_list_method1[robot].append(mean_val_method1)
    #         error_margin_method1[robot].append(std_val_method1)

    #         # Method 2 (assuming self.SR_2 contains data for method 2)
    #         mean_val_method2 = np.mean(chamfer_score_nv_2[(robot)])  # Assuming this is for method 2
    #         std_val_method2 = np.std(chamfer_score_nv_2[(robot)])
    #         robot_list_method2[robot].append(mean_val_method2)
    #         error_margin_method2[robot].append(std_val_method2)   
                         
    # robots = ['Shadow', 'Allegro', 'Barrett', 'Robotiq']
    # labels = ['5-finger', '4-finger', '3-finger', '2-finger']
    # x = np.arange(len(robots))

    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 3.5), dpi=300,
    #                                gridspec_kw={'height_ratios': [2, 1]})
    # fig.subplots_adjust(hspace=0.05)  # Adjust space between axes
    
    # # Plot Method 2 (Baseline) on ax1 (lower range)
    # method2_mean = [robot_list_method2[robot][0] for robot in robots]
    # method2_error = [error_margin_method2[robot][0] for robot in robots]

    # ax1.plot(x, method2_mean, label="Baseline", color='#CA6702', linewidth=2)
    # ax1.fill_between(x, np.array(method2_mean) - np.array(method2_error), np.array(method2_mean) + np.array(method2_error), color='#CA6702', alpha=0.2, edgecolor=None)
    
    # # ax1.set_ylabel('Net Wrench Residual (Baseline)', color='#CA6702')
    # ax1.tick_params(axis='y', labelcolor='#CA6702')
    # ax1.set_xticklabels([])  # Remove x-ticklabels from the upper axis
    # # ax1.set_xlabel('')
    # # ax1.tick_params(axis='x', labelbottom=False)
    # # Plot Method 1 (Ours) on ax2 (higher range)
    # method1_mean = [robot_list_method1[robot][0] for robot in robots]
    # method1_error = [error_margin_method1[robot][0] for robot in robots]

    # ax2.plot(x, method1_mean, label="Ours", color='#005F73', linewidth=2)
    # ax2.fill_between(x, np.array(method1_mean) - np.array(method1_error), np.array(method1_mean) + np.array(method1_error), color='#005F73', alpha=0.2, edgecolor=None)
    
    # # ax2.set_ylabel('Net Wrench Residual (Ours)', color='#005F73')
    # ax2.tick_params(axis='y', labelcolor='#005F73')
    # ax1.legend(loc='best')
    # ax2.legend(loc='best')
    # # Set x-axis labels
    # ax2.set_xticks(x)
    # ax2.set_xticklabels(labels)

    # # Set the limits for the axes
    # ax1.set_ylim(3, 35)  # Higher values (Baseline)
    # ax2.set_ylim(0, 2)  # Lower values (Ours)

    # # Add a gap between the axes to simulate a broken axis
    # # ax1.set_xticklabels([])  # Remove x-ticklabels from the upper axis
    # # ax1.set_xlabel('')
    # # ax1.set_xticks(x)

    # # ax2.set_xlabel('Robot Types')
    # ax1.spines.bottom.set_visible(False)
    # ax2.spines.top.set_visible(False)
    # ax1.xaxis.set_ticks_position('none')  # no ticks on the upper axis

    # ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    # ax2.xaxis.tick_bottom()
    # plt.setp(ax2.get_yticklabels()[2], visible=False)  # Hide the last y-tick label
    # plt.setp(ax1.get_yticklabels()[0], visible=False)  # Hide the last y-tick label
    # plt.setp(ax2.get_yticklines()[4], visible=False)  # Hide the last y-tick label
    # plt.setp(ax1.get_yticklines()[0], visible=False)  # Hide the last y-tick label
    # # Add a global title for the whole figure
    # fig.suptitle('Net wrench residual over Robots (mu = 0.9)', fontsize=12)
    # fig.text(0.01, 0.5, 'Net Wrench Residual', ha='center', va='center', rotation='vertical', fontsize=12)

    # # Add a line to indicate the break
    # d = .5  # Proportion of vertical to horizontal extent of the slanted line
    # kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
    #               linestyle="none", color='k', mec='k', mew=1, clip_on=False)

    # # Slanted lines between ax1 and ax2
    # ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    # ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    # fig.tight_layout()

    # plt.savefig("SR_mu_comparison_0.9_Net_wrench_residual.png", dpi=300, facecolor='#FFFFFF', edgecolor='none')

    # plt.show()

    ############################################################################
    
    # fig, ax = plt.subplots(figsize=(7, 3.5), dpi=300)

    # # Plot Method 1 as a line with shaded error band
    # method1_mean = [robot_list_method1[robot][0] for robot in robots]
    # method1_error = [error_margin_method1[robot][0] for robot in robots]

    # ax.plot(x, method1_mean, label="Ours", color='#005F73', linewidth=2)
    # ax.fill_between(x, np.array(method1_mean) - np.array(method1_error), np.array(method1_mean) + np.array(method1_error), color='#005F73', alpha=0.2, edgecolor=None)

    # # Plot Method 2 as a line with shaded error band
    # method2_mean = [robot_list_method2[robot][0] for robot in robots]
    # method2_error = [error_margin_method2[robot][0] for robot in robots]

    # ax.plot(x, method2_mean, label="Baseline", color='#CA6702', linewidth=2)
    # ax.fill_between(x, np.array(method2_mean) - np.array(method2_error), np.array(method2_mean) + np.array(method2_error), color='#CA6702', alpha=0.2, edgecolor=None)

    # # Set labels and title
    # ax.set_ylabel('Net wrench residual')
    # ax.set_title('Net wrench residual over Robots (mu = 0.9)')
    # ax.set_xticks(x)  # Set the x-ticks at the robot positions
    # ax.set_xticklabels(labels)  # Use robot names as x-axis labels
    # ax.legend(loc='best')

    # fig.tight_layout()
    # plt.savefig("SR_mu_comparison_0.9_Net wrench_residual.png", dpi=300, facecolor='#FFFFFF', edgecolor='none')

    # plt.show()
            #chamfer_score_nv[(robot)] = residual_tensor.argmax()
    #print(chamfer_score_nv)
    #####################################################################################
    # chamfer_score_nv = {}
    # robot = 'Robotiq'
    # mu = 0.9
    # sim = similarity(robot, mu, device='cuda:1', exp_name='genhand')
    # chamfer_score_nv[(robot)] = sim.run_sdf_residual_test()
    # print(chamfer_score_nv)
    
    # robots = ['Shadow', 'Allegro', 'Barrett', 'Robotiq']
    # labels = ['ours', 'dexycb']
    # robot_data = {robot: [np.mean(chamfer_score[robot]), np.mean(chamfer_score_nv([robot]))] for robot in robots}
    # error_data = {robot: [stats.sem(chamfer_score[robot]),stats.sem(chamfer_score_nv[robot])] for robot in robots}
    # x = np.arange(len(robots))  # X positions for robots
    # width = 0.3  # Bar width
    # fig, ax = plt.subplots(figsize=(7, 7/2), dpi=300)
    # color = ['#005F73']
    # # Plot bars for each time category
    # for i, label in enumerate(labels):
    #     cham = [robot_data[robot][i] for robot in robots]
    #     error = [error_data[robot][i] for robot in robots]
    #     ax.bar(x + i * width, cham, width, label=label, yerr=error, capsize=5, color=colors[i])
    # ax.set_ylabel('Chamfer Distance')
    # ax.set_title('Chamfer Distance over Robots')
    # ax.set_xticks(x + width / 2)  # Center the tick labels
    # ax.set_xticklabels(robots)  # Set x-axis labels to robot names
    # fig.tight_layout()
    # plt.ylim(0, max([np.mean(chamfer_score[robot]) for robot in robots]) + max([stats.sem(chamfer_score[robot]) for robot in robots]) + 2)
    # plt.savefig("chamfer.png", dpi=300, facecolor='#FFFFFF', edgecolor='none')
    # plt.show()
    # # sim = similarity('Shadow', 0.7, exp_name='nv', device='cuda:0')
    # # print(sim.run_similarity())
def plot_success_rate_with_error():
    # Generate some random data for testing
    robots = ['Shadow', 'Allegro', 'Barrett', 'Robotiq']
    labels = ['5-finger', '4-finger', '3-finger', '2-finger']
    x = np.arange(len(robots))
    
    # Simulating some data for method1 and method2 for illustration
    robot_list_method1 = {'Shadow': [0.5], 'Allegro': [0.3], 'Barrett': [0.7], 'Robotiq': [0.6]}
    error_margin_method1 = {'Shadow': [0.1], 'Allegro': [0.1], 'Barrett': [0.1], 'Robotiq': [0.1]}
    
    robot_list_method2 = {'Shadow': [12], 'Allegro': [15], 'Barrett': [18], 'Robotiq': [16]}
    error_margin_method2 = {'Shadow': [2], 'Allegro': [2], 'Barrett': [2], 'Robotiq': [2]}

    # Create the figure and the first axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 5), dpi=300,
                                   gridspec_kw={'height_ratios': [2.5, 1]})
    fig.subplots_adjust(hspace=0.05)  # Adjust space between axes
    
    # Plot Method 2 (Baseline) on ax1 (lower range)
    method2_mean = [robot_list_method2[robot][0] for robot in robots]
    method2_error = [error_margin_method2[robot][0] for robot in robots]

    ax1.plot(x, method2_mean, label="Baseline", color='#CA6702', linewidth=2)
    ax1.fill_between(x, np.array(method2_mean) - np.array(method2_error), np.array(method2_mean) + np.array(method2_error), color='#CA6702', alpha=0.2, edgecolor=None)
    
    # ax1.set_ylabel('Net Wrench Residual (Baseline)', color='#CA6702')
    ax1.tick_params(axis='y', labelcolor='#CA6702')
    ax1.set_xticklabels([])  # Remove x-ticklabels from the upper axis
    # ax1.set_xlabel('')
    # ax1.tick_params(axis='x', labelbottom=False)
    # Plot Method 1 (Ours) on ax2 (higher range)
    method1_mean = [robot_list_method1[robot][0] for robot in robots]
    method1_error = [error_margin_method1[robot][0] for robot in robots]

    ax2.plot(x, method1_mean, label="Ours", color='#005F73', linewidth=2)
    ax2.fill_between(x, np.array(method1_mean) - np.array(method1_error), np.array(method1_mean) + np.array(method1_error), color='#005F73', alpha=0.2, edgecolor=None)
    
    # ax2.set_ylabel('Net Wrench Residual (Ours)', color='#005F73')
    ax2.tick_params(axis='y', labelcolor='#005F73')
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    # Set x-axis labels
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)

    # Set the limits for the axes
    ax1.set_ylim(3, 30)  # Higher values (Baseline)
    ax2.set_ylim(0, 0.8)  # Lower values (Ours)

    # Add a gap between the axes to simulate a broken axis
    # ax1.set_xticklabels([])  # Remove x-ticklabels from the upper axis
    # ax1.set_xlabel('')
    # ax1.set_xticks(x)

    # ax2.set_xlabel('Robot Types')
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.set_ticks_position('none')  # no ticks on the upper axis

    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    plt.setp(ax2.get_yticklabels()[2], visible=False)  # Hide the last y-tick label
    plt.setp(ax1.get_yticklabels()[0], visible=False)  # Hide the last y-tick label
    plt.setp(ax2.get_yticklines()[4], visible=False)  # Hide the last y-tick label
    plt.setp(ax1.get_yticklines()[0], visible=False)  # Hide the last y-tick label
    # Add a global title for the whole figure
    fig.suptitle('Net Wrench Residual Comparison: Ours vs Baseline (mu = 0.9)', fontsize=12)
    fig.text(0.015, 0.5, 'Net Wrench Residual', ha='center', va='center', rotation='vertical', fontsize=12)

    # Slanted lines to indicate the break (creating the "cut-out" effect)
    d = .5  # Proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    # Slanted lines between ax1 and ax2
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    # Adjust the layout
    fig.tight_layout()
    # plt.subplots_adjust(hspace=0.1)  # Control the space between axes

    # Show the plot
    plt.show()


    # Plot Method 1 on ax1 (lower range)

def draw_dist_residual_single():
    gh_record = 'distance_residual_gh.json'
    with open(osp.join(osp.dirname(__file__),gh_record), 'r') as f:
        gh_data = json.load(f)

    
    robot_list_gh = {}
    error_margin_gh = {}
    for mu in [0.1,0.3,0.5,0.7,0.9]:
        robot_list_gh[mu] = []
        error_margin_gh[mu] = []
        for robot in ['Robotiq', 'Barrett', 'Allegro', 'Shadow']:
            mean_val_gh = np.mean(gh_data[f"{robot, mu}"])*100
            std_val_gh = stats.sem(gh_data[f"{robot, mu}"])*100
            robot_list_gh[mu].append(mean_val_gh)
            error_margin_gh[mu].append(std_val_gh)
    print("robot_list_gh", robot_list_gh)
    # robots = ['Shadow', 'Allegro', 'Barrett', 'Robotiq']
    robots = ['Robotiq', 'Barrett', 'Allegro', 'Shadow']
    mus = [0.1,0.3,0.5,0.7,0.9]
    # labels = ['Shadow\n5-finger', 'Allegro\n4-finger', 'Barrett\n3-finger', 'Robotiq\n2-finger']
    labels = ['Robotiq\n2-finger', 'Barrett\n3-finger', 'Allegro\n4-finger', 'Shadow\n5-finger']
    x = np.arange(len(robots))
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=300)

    maker = ['o', '^', 's', 'D', 'P']
    color = ['#005F73', '#CA6702']

    for j, mu in enumerate(mus):
        mean_val_gh = robot_list_gh[mu]
        error_val_gh = error_margin_gh[mu]
        ax.plot(x, mean_val_gh, label=r'$\mu=$'+str(mu), color=color[0], marker=maker[j], markersize=7, linewidth=1)
        ax.fill_between(x, np.array(mean_val_gh) - np.array(error_val_gh), np.array(mean_val_gh) + np.array(error_val_gh), color=color[0], alpha=0.2, edgecolor=None)

    ax.set_ylabel('Distance value residual (cm)')
    ax.set_title('Distance value residual over Robots')
    ax.set_xticks(x)  # Set the x-ticks at the robot positions
    ax.set_xticklabels(labels)  # Use robot names as x-axis labels
    ax.legend(loc='best')


    fig.tight_layout()

    plt.savefig("SR_mu_comparison_dist_residual_single.png", dpi=300, facecolor='#FFFFFF', edgecolor='none')

    plt.show()

if __name__ == '__main__':
    # plot_success_rate_with_error()
    # test_dataset()
    # draw_dist_residual_single()
    # evaluate()
    # test_opt()
    draw_gf_residual()
    """
    Note GenHand:
    {('Shadow', 0.1): tensor(4.0793, dtype=torch.float64), 
    ('Shadow', 0.3): tensor(4.1053, dtype=torch.float64), 
    ('Shadow', 0.5): tensor(4.1060, dtype=torch.float64), 
    ('Shadow', 0.7): tensor(4.1099, dtype=torch.float64), 
    ('Shadow', 0.9): tensor(4.0908, dtype=torch.float64), 
    ('Allegro', 0.1): tensor(4.5979, dtype=torch.float64), 
    ('Allegro', 0.3): tensor(4.6089, dtype=torch.float64), 
    ('Allegro', 0.5): tensor(4.5955, dtype=torch.float64), 
    ('Allegro', 0.7): tensor(4.5816, dtype=torch.float64),
    ('Allegro', 0.9): tensor(4.5467, dtype=torch.float64), 
    ('Barrett', 0.1): tensor(4.7647, dtype=torch.float64), 
    ('Barrett', 0.3): tensor(4.7732, dtype=torch.float64), 
    ('Barrett', 0.5): tensor(4.7772, dtype=torch.float64), 
    ('Barrett', 0.7): tensor(4.7805, dtype=torch.float64), 
    ('Barrett', 0.9): tensor(4.7646, dtype=torch.float64), 
    {('Robotiq', 0.1): tensor(5.4060, dtype=torch.float64), 
    ('Robotiq', 0.3): tensor(5.3954, dtype=torch.float64), 
    ('Robotiq', 0.5): tensor(5.4133, dtype=torch.float64), 
    ('Robotiq', 0.7): tensor(5.3617, dtype=torch.float64), 
    ('Robotiq', 0.9): tensor(5.4065, dtype=torch.float64)}


    NV:
    {('Shadow', 0.1): tensor(4.1046, dtype=torch.float64), 
    ('Shadow', 0.3): tensor(4.1013, dtype=torch.float64), 
    ('Shadow', 0.5): tensor(4.1000, dtype=torch.float64), 
    ('Shadow', 0.7): tensor(4.1353, dtype=torch.float64), 
    ('Shadow', 0.9): tensor(4.0808, dtype=torch.float64), 
    ('Allegro', 0.1): tensor(4.5830, dtype=torch.float64), 
    ('Allegro', 0.3): tensor(4.5981, dtype=torch.float64), 
    ('Allegro', 0.5): tensor(4.6000, dtype=torch.float64), 
    ('Allegro', 0.7): tensor(4.5877, dtype=torch.float64), 
    ('Allegro', 0.9): tensor(4.5715, dtype=torch.float64), 
    ('Barrett', 0.1): tensor(4.7488, dtype=torch.float64), 
    ('Barrett', 0.3): tensor(4.7637, dtype=torch.float64), 
    ('Barrett', 0.5): tensor(4.7144, dtype=torch.float64), 
    ('Barrett', 0.7): tensor(4.7794, dtype=torch.float64), 
    ('Barrett', 0.9): tensor(4.7628, dtype=torch.float64), 
    {('Robotiq', 0.1): tensor(5.3792, dtype=torch.float64),
    ('Robotiq', 0.3): tensor(5.3697, dtype=torch.float64), 
    ('Robotiq', 0.5): tensor(5.3394, dtype=torch.float64), 
    ('Robotiq', 0.7): tensor(5.3931, dtype=torch.float64), 
    ('Robotiq', 0.9): tensor(5.3706, dtype=torch.float64)}



    """
    # import numpy as np
    # import matplotlib.pyplot as plt

    # # 🔹 Define 5 categories (Y-axis labels)
    # categories = ['0.1', '0.3', '0.5', '0.7', '0.9']

    # # 🔹 Number of bars per category
    # num_bars = 8

    # # 🔹 Generate random data (8 bars for each of the 5 categories)
    # # data = np.random.rand(len(categories), num_bars) * 10  # Shape (5, 8)
    
    # baseset = {'Shadow': [0.10309278350515463, 0.31958762886597936, 0.4791666666666667, 0.7938144329896907, 0.845360824742268],
    #            'Allegro': [0.09511568123393316, 0.28974358974358977, 0.5461538461538461, 0.7628865979381443, 0.8560411311053985],
    #            'Barrett': [0.09948979591836735, 0.32908163265306123, 0.5586734693877551, 0.7061855670103093, 0.8132992327365729],
    #            'Robotiq': [0.0663265306122449, 0.17857142857142858, 0.5612244897959183, 0.6930946291560103, 0.7346938775510204],
    #             'Shadow_nv':  [0.07216494845360824, 0.2708333333333333, 0.3917525773195876, 0.59375, 0.6391752577319587],
    #             'Allegro_nv': [0.053164556962025315, 0.17721518987341772, 0.3848101265822785, 0.6227848101265823, 0.748730964467005],
    #             'Barrett_nv': [0.11645569620253164, 0.369620253164557, 0.5645569620253165, 0.7316455696202532, 0.8075949367088607],
    #             'Robotiq_nv': [0.002544529262086514, 0.04580152671755725, 0.089058524173028, 0.15012722646310434, 0.22391857506361323]
    # }
    
    # data = []
    # for i in range(5):
    #     data.append([baseset['Shadow'][i],  baseset['Shadow_nv'][i], 
    #                  baseset['Allegro'][i], baseset['Allegro_nv'][i],
    #                  baseset['Barrett'][i], baseset['Barrett_nv'][i],
    #                  baseset['Robotiq'][i], baseset['Robotiq_nv'][i]])
    # data = np.array(data)
    
    # # 🔹 Bar width & Y-axis positions
    # y = np.arange(len(categories))  # Positions for categories
    # bar_width = 0.1  # Controls spacing between bars

    # # 🔹 Colors for each bar group
    # colors = plt.cm.get_cmap('tab10', num_bars).colors  # Unique color for each bar

    # # 🔹 Create Figure
    # fig, ax = plt.subplots(figsize=(10, 6))

    # # 🔹 Plot each group of 8 bars
    # for i in range(num_bars):
    #     ax.barh(y + (i - num_bars/2) * bar_width, data[:, i], height=bar_width, color=colors[i], label=f'Bar {i+1}')

    # # 🔹 Formatting
    # ax.set_yticks(y)
    # ax.set_yticklabels(categories)
    # ax.set_xlabel("Frictional Coefficient")
    # ax.set_title("Success rate over Frictional Coefficient")
    # ax.legend(ncol=4, fontsize=8, loc="upper right")

    # plt.show()

    """
    single:
        {'Shadow': [nan, nan, nan, nan, 0.6979166666666666], 'Allegro': [nan, nan, nan, nan, 0.6701030927835051], 'Barrett': [nan, nan, nan, nan, 0.6597938144329897], 'Robotiq': [nan, nan, nan, nan, 0.6082474226804123]}

    genhand contact:
    {'Robotiq': tensor(0.0105), 'Shadow': tensor(0.0031), 'Allegro': tensor(0.0031), 'Barrett': tensor(0.0028)}
    
    {'Robotiq': tensor(0.0156), 'Shadow': tensor(0.0181), 'Allegro': tensor(0.0182), 'Barrett': tensor(0.0139)}
    gf
    {'Robotiq': tensor(4.4439), 'Shadow': tensor(26.7702), 'Allegro': tensor(16.8093), 'Barrett': tensor(10.1775)}
   
    {'Robotiq': tensor(0.1789), 'Shadow': tensor(0.4492), 'Allegro': tensor(0.4206), 'Barrett': tensor(0.2972)}

    
      """
