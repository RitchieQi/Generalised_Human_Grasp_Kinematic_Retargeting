import os
osp = os.path
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc
import numpy as np
from typing import List, Optional, Tuple, Union, Iterator, Any, Dict
from dex_ycb_toolkit.dex_ycb import DexYCBDataset
import json
from scipy.spatial.transform import Rotation as R
import torch
import time
import trimesh as tm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import cv2
from manopth.manolayer import ManoLayer
from tqdm import tqdm
import traceback
from multiprocessing import Process, Queue
from itertools import product
import os
from bullet_base import PybulletBase
from object_base import objects
from robot_base import shadow, allegro, barrett, robotiq, MountingBase
from opt_results_loader import ycb_opt_fetcher
from trajectory import Trajectory
from optimisation.utils import rotate_vector, rotate_vector_inverse, quat_mult
class Environment:
    def __init__(self,
                 robot: str = "Shadow",
                 mu: float = 0.9,
                 exp: str = "genhand",
                 repeat: int = 1,
                 render: str = "GUI",
                 data_sample: int = 20,
                 task: str = "test",
                 test_sdf: bool = False,
                 ) -> None:
        self.sim = PybulletBase(connection_mode=render)
        if robot == "Shadow":
            self.robot = shadow(self.sim)
        elif robot == "Allegro":
            self.robot = allegro(self.sim)
        elif robot == "Barrett":
            self.robot = barrett(self.sim)
        elif robot == "Robotiq":
            self.robot = robotiq(self.sim)
        self.robot.set_robot_info()
        self.object = objects(self.sim)
        self.base = MountingBase(self.sim, self.robot)
        self.mu = mu
        self.data_fetcher = ycb_opt_fetcher(robot_name=self.robot.body_name,
                                            mu = mu, 
                                            exp_name=exp, 
                                            repeat=repeat, 
                                            data_sample=data_sample, 
                                            task=task, 
                                            test_sdf=test_sdf)
        # self.data_fetcher = ycb_opt_fetcher(robot_name=self.robot.body_name,mu = mu, exp_name=exp, repeat=repeat, data_sample=None, task="test")

        self.re_orent = R.from_matrix(np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])).as_quat()
        self.pre_grasp = self.robot.pre_grasp
        self.rob_joints = None
        self.obj_quat = None
        self.rob_glo_quat = None
        self.rob_glo_trans = None
        self.obj_quat = None
        self.obj_trans = None
        self.lift_up = np.array([0.0, 0.0, 0.3])

        if task == "test" and test_sdf is False:
            if exp == "genhand":
                self.record_dir = osp.join(osp.dirname(__file__), 'record') 
                self.sim_result_dir = osp.join(osp.dirname(__file__), 'sim_result')
            elif exp == "nv":
                self.record_dir = osp.join(osp.dirname(__file__), 'record_nv') 
                self.sim_result_dir = osp.join(osp.dirname(__file__), 'sim_result_nv')
                self.image_dir = osp.join(osp.dirname(__file__), 'image_nv')
            else:
                self.record_dir = osp.join(osp.dirname(__file__), 'record_test')
                self.sim_result_dir = osp.join(osp.dirname(__file__), 'sim_result_test')
            if not osp.exists(self.record_dir):
                os.makedirs(self.record_dir)
            if not osp.exists(self.sim_result_dir):
                os.makedirs(self.sim_result_dir)
        if task == "test" and test_sdf is True:
            if exp == "genhand":
                self.record_dir = osp.join(osp.dirname(__file__), 'record_sdf') 
                self.sim_result_dir = osp.join(osp.dirname(__file__), 'sim_result_sdf')
            elif exp == "nv":
                self.record_dir = osp.join(osp.dirname(__file__), 'record_sdf_nv') 
                self.sim_result_dir = osp.join(osp.dirname(__file__), 'sim_result_sdf_nv')
                self.image_dir = osp.join(osp.dirname(__file__), 'image_sdf_nv')
            else:
                self.record_dir = osp.join(osp.dirname(__file__), 'record_test')
                self.sim_result_dir = osp.join(osp.dirname(__file__), 'sim_result_test')
            if not osp.exists(self.record_dir):
                os.makedirs(self.record_dir)
            if not osp.exists(self.sim_result_dir):
                os.makedirs(self.sim_result_dir)            

        self.image_dir = osp.join(osp.dirname(__file__), 'image')
        if not osp.exists(self.image_dir):
            os.makedirs(self.image_dir)
    def save_image(self, rgb, name):
        cv2.imwrite(osp.join(self.image_dir, name), rgb)
    
    def initialisation(self, data_pack) -> None:
        # data_pack = self.data_fetcher[idx]
        obj_quat = data_pack["obj_quat"]
        obj_trans = data_pack["obj_trans"]
        obj_centre = data_pack["obj_centre"]
        rob_mat = data_pack["rob_mat"]
        rob_trans = data_pack["rob_trans"]
        self.rob_joints = data_pack["joint_val"]
        self.obj_file = data_pack["obj_file"]
        print("joint values",self.rob_joints)
        rob_glo_quat, rob_glo_trans = self.robot.pre_transformation(rob_mat, rob_trans)
        self.rob_glo_quat = quat_mult(self.re_orent, rob_glo_quat)
        self.obj_quat = quat_mult(self.re_orent, obj_quat)
        self.rob_glo_trans = rotate_vector(rob_glo_trans, self.re_orent)
        self.obj_trans = rotate_vector(obj_trans-obj_centre, self.re_orent)
        self.pre_grasp_ = rotate_vector(self.pre_grasp, self.rob_glo_quat)
        
        self.traj = Trajectory(initial_positions=np.zeros(3), 
                          initial_joint=self.robot.init_joint, 
                          grasp_position=-1*self.robot.pre_grasp, 
                          grasp_joint=self.rob_joints, 
                          lift_up=rotate_vector_inverse((self.lift_up-self.pre_grasp_), self.rob_glo_quat))
        #self.record_name = self.robot.body_name + "_" + str(self.mu) + "_" + str(idx) + ".mp4"
    
    def load(self) -> None:
        self.object.set_object_info(file_path=self.obj_file, 
                                    position=self.obj_trans, 
                                    orientation=self.obj_quat, 
                                    obj_mass=0.1)
        self.object.load_object(mu=self.mu)
        self.sim.load_plane()
        #self.object.create_world_contraint()
        self.robot.load_robot(base_position=self.rob_glo_trans+self.pre_grasp_, 
                              base_orientation=self.rob_glo_quat, 
                              joints_val=self.robot.init_joint, 
                              lateral_mu = self.mu, 
                              Fixbase=False)
        self.mount, _ = self.base.mounting_gripper()
        #self.robot.disable_motor()
    
    def load_grsap_position(self) -> None:
        self.sim.load_plane()
        self.robot.load_robot(base_position=self.rob_glo_trans,
                                base_orientation=self.rob_glo_quat,
                                joints_val=self.rob_joints,
                                lateral_mu=self.mu,
                                Fixbase=False)
        self.mount, _ = self.base.mounting_gripper()
        self.object.set_object_info(file_path=self.obj_file,
                                    position=self.obj_trans,
                                    orientation=self.obj_quat,
                                    obj_mass=0.1)
    
    def image_screenshot(self) -> None:
        self.robot.disable_motor()
        self.object.create_world_contraint()
        for t in range(2):
            self.sim.pc.stepSimulation()
            if t == 1:
                image = self.sim.get_camera_image()
        return image

    def start_image(self) -> None:
        qb, vb, ab, tb, qj, vj, aj, tj = self.traj.trajectory_generate_v2()
        self.robot.disable_motor()
        # self.object.create_mount_contraint()
        self.object.create_world_contraint()
        for t in range(1000*self.traj.get_reach_duration):

            for i in range(len(self.sim.control_joint)):
                self.sim.pc.setJointMotorControl2(
                                    bodyIndex=self.sim._bodies_idx[self.robot.body_name],
                                    jointIndex=self.sim.control_joint[i], 
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=qj[t, i],
                                    )                
            
            for i in range(self.base.get_num_joints()):
                self.sim.pc.setJointMotorControl2(
                bodyIndex=self.mount,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=qb[t, i],
                # targetVelocity=v_0[t, i],
            )
            if t == 1000*self.traj.get_reach_duration-1:
                image = self.sim.get_camera_image()
                # time.sleep(3.0)
            self.sim.pc.stepSimulation()
            
            
        return image

                
    def start(self, record) -> None:
        qb, vb, ab, tb, qj, vj, aj, tj = self.traj.trajectory_generate_v2()
        self.robot.disable_motor()
        # self.object.create_mount_contraint()
        self.object.create_world_contraint()

        if record:
            self.sim.start_record(osp.join(self.record_dir, self.record_name))
        for t in range(1000*self.traj.get_reach_duration):
            for i in range(len(self.sim.control_joint)):
                self.sim.pc.setJointMotorControl2(
                                    bodyIndex=self.sim._bodies_idx[self.robot.body_name],
                                    jointIndex=self.sim.control_joint[i], 
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=qj[t, i],
                                    )                
            
            for i in range(self.base.get_num_joints()):
                self.sim.pc.setJointMotorControl2(
                bodyIndex=self.mount,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=qb[t, i],
                # targetVelocity=v_0[t, i],
            )
            self.sim.pc.stepSimulation()
            if record:
                time.sleep(1.0 / 1000.0)
        
        self.object.remove_world_contraint()
        # rgb = self.sim.get_camera_image()
        # time.sleep(3.0)

        for t in range(1000*self.traj.get_reach_duration, 1000*self.traj.get_duration):
            for i in range(len(self.sim.control_joint)):
                self.sim.pc.setJointMotorControl2(
                                    bodyIndex=self.sim._bodies_idx[self.robot.body_name],
                                    jointIndex=self.sim.control_joint[i], 
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=qj[t, i],
                                    ) 
                
            for i in range(self.base.get_num_joints()):
                    self.sim.pc.setJointMotorControl2(
                    bodyIndex=self.mount,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=qb[t, i],
                )
            if t == 1000*self.traj.get_duration-1:
                contacts = self.sim.pc.getContactPoints(self.sim._bodies_idx[self.robot.body_name], self.sim._bodies_idx[self.object.body_name])
                if contacts:
                    contact_link = []
                    contact_position = []
                    contact_dist = []
                    for contact in contacts:
                        #print("contact", contact[8])
                        contact_dist.append(contact[8])
                        contact_link.append(contact[3])
                        contact_position.append(contact[6])
                        # contact_info = dict(contact_link = contact[3], contact_position = contact[6])
                    contact_info = dict(flag = True, contact_link = contact_link, contact_position = contact_position, contact_dist = contact_dist)
                
                else:
                    contact_info = dict(flag = False, contact_link = [], contact_position = [], contact_dist = [])


            self.sim.pc.stepSimulation()
            if record:
                time.sleep(1.0 / 1000.0)
            
        print("contact_info", contact_info["flag"])
        if record:
            self.sim.stop_record()
        return contact_info#, rgb
            

    def run_image(self, idx: int) -> None:
        self.sim._connect_()
        try:
            self.initialisation(self.data_fetcher[idx])
            self.load()
            image = self.start_image()
            self.remove_scene()
            self.image_name = self.robot.body_name + "_" + str(self.mu) + "_" + str(idx) + ".png"

            self.save_image(image, self.image_name)
        except Exception as e:
            print(e)
            traceback.print_exc()
            self.remove_scene()
    
        self.sim.pc.disconnect()

    def screen_shot(self, idx: int) -> None:
        self.sim._connect_()
        try:
            self.initialisation(self.data_fetcher[idx])
            self.load_grsap_position()
            image = self.image_screenshot()
            self.image_name = self.robot.body_name + "_" + str(self.mu) + "_" + str(idx) + ".png"
            self.save_image(image, self.image_name)
        except Exception as e:
            print(e)
            traceback.print_exc()
            self.remove_scene()
    
    def run_idx(self, idx: int, record: bool) -> None:
        self.sim._connect_()

        try:
            self.initialisation(self.data_fetcher[idx])
            self.record_name = self.robot.body_name + "_" + str(self.mu) + "_" + str(idx) + ".mp4"
            self.load()
            contact_info = self.start(record)
            self.remove_scene()
            #self.save_image(im, self.image_name)
        except Exception as e:
            print(e)
            traceback.print_exc()
            contact_info = dict(flag = "Error", contact_link = [], contact_position = [])
            self.remove_scene()
        file_name = self.robot.body_name + "_" + str(self.mu) + "_" + str(idx) + ".json"
        with open(osp.join(self.sim_result_dir,file_name), 'w') as f:
            json.dump(contact_info, f, indent=4)
    
        self.sim.pc.disconnect()

    def remove_scene(self) -> None:
        self.sim.remove_all()
        self.sim.reset_simulation()
        
    def run(self, record) -> None:
        # loader = DataLoader(self.data_fetcher, batch_size=1, shuffle=False)
        
        bar = tqdm(enumerate(self.data_fetcher), total=len(self.data_fetcher))
        for idx, data_pack in enumerate(bar):
            self.sim._connect_()
            try:
                self.initialisation(data_pack[1])
                self.record_name = self.robot.body_name + "_" + str(self.mu) + "_" + str(idx) + ".mp4"
                self.image_name = self.robot.body_name + "_" + str(self.mu) + "_" + str(idx) + ".png"

                self.load()
                contact_info = self.start(record)
                self.remove_scene()
                #self.save_image(im, self.image_name)
            except Exception as e:
                print(e)
                traceback.print_exc()
                contact_info = dict(flag = "Error", contact_link = [], contact_position = [], contact_dist = [])
                self.remove_scene()

            file_name = self.robot.body_name + "_" + str(self.mu) + "_" + str(idx) + ".json"
            with open(osp.join(self.sim_result_dir,file_name), 'w') as f:
                json.dump(contact_info, f, indent=4)
        
            self.sim.pc.disconnect()
