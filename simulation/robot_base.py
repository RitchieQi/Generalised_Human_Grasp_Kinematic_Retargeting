import os
osp = os.path
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc
import numpy as np
from typing import List, Optional, Tuple, Union, Iterator, Any, Dict
from dex_ycb_toolkit.dex_ycb import DexYCBDataset
from scipy.spatial.transform import Rotation as R
from abc import ABC, abstractmethod
import os
from bullet_base import PybulletBase
from optimisation_tmp.utils import rodrigues, euler_to_mat
import torch

class robot(ABC):
    def __init__(
        self,
        sim: PybulletBase,

    )-> None:
        self.sim = sim
        self.body_name = None
        self.urdf_path = None
        # self.base_position = None
        # self.base_orientation = None
        self.endeffect = None
        self.init_joint = None
        self.calibration_rot = None
        self.calibration_trans = None
        self.pre_grasp = None
        #self.set_robot_info()
        #self.load_robot()
         
    @abstractmethod
    def set_robot_info(self) -> None:
        pass
    
    def load_robot(self, base_position, base_orientation, joints_val: np.ndarray, lateral_mu: float, Fixbase: int) -> None:
        if self.urdf_path is None:
            self.set_robot_info()
        
        self.sim.loadURDF(
            body_name=self.body_name,
            fileName=self.urdf_path,
            basePosition=base_position,
            baseOrientation=base_orientation,
            useFixedBase=Fixbase,
        )
        self.sim.set_joint_angles_dof(self.body_name, angles=joints_val)
        # self.sim.set_lateral_friction(self.body_name, link=self.endeffect, lateral_friction=lateral_mu)
        # self.sim.set_spinning_friction(self.body_name, link=self.endeffect, spinning_friction=lateral_mu)
        # self.sim.set_rolling_friction(self.body_name, link=self.endeffect, rolling_friction=lateral_mu)
        self.sim.set_friction_all(self.body_name, lateral_friction=lateral_mu)
        self.sim.setup_dynamics(self.body_name)
        #self.sim.set_joint_angles_dof(self.body_name, angles=self.init_joint)
    
    def reset(self, position: np.ndarray, orientation: np.ndarray, joints_val: np.ndarray, lateral_mu: float) -> None:
        self.sim.set_base_pose(self.body_name, position, orientation)

        self.sim.set_joint_angles_dof(self.body_name, angles=joints_val)
        self.sim.set_lateral_friction(self.body_name, link=self.endeffect, lateral_friction=lateral_mu)
    
    def pre_transformation(self, rot: np.ndarray, trans: np.ndarray):

        R_f = np.matmul(rot, self.calibration_rot)
        T_f = -np.matmul(rot, np.matmul(self.calibration_rot, self.calibration_trans.reshape(3,1))) + trans.reshape(3,1)

        global_rot = R.from_matrix(R_f).as_quat()
        global_trans = T_f.reshape(3)
        return global_rot, global_trans
    
    def step(self) -> None:
        self.sim.step()
    
    def get_link_position(self, link: int) -> np.ndarray:
        """Returns the position of a link as (x, y, z)

        Args:
            link (int): The link index.

        Returns:
            np.ndarray: Position as (x, y, z)
        """
        return self.sim.get_link_position(self.body_name, link)
    
    def get_joint_angle(self, joint: int) -> float:
        """Returns the angle of a joint

        Args:
            joint (int): The joint index.

        Returns:
            float: Joint angle
        """
        return self.sim.get_joint_angle(self.body_name, joint)
    
    def get_joint_angles_dof(self) -> np.ndarray:
        """Returns the angles of all joints.

        Returns:
            np.ndarray: Joint angles
        """
        return self.sim.get_joint_angles_dof(self.body_name)

    def get_joint_velocities_dof(self) -> np.ndarray:
        """Returns the velocities of all joints.

        Returns:
            np.ndarray: Joint velocities
        """
        return self.sim.get_joint_velocities_dof(self.body_name)

    def set_joint_angles(self, angles: list) -> None:
        """Set the joint position of a body. Can induce collisions.

        Args:
            angles (list): Joint angles.
        """
        self.sim.set_joint_angles(self.body_name, joints=self.joint_indices, angles=angles)
        
    def disable_motor(self) -> None:
        "disable motor"
        self.sim.disable_motor(self.body_name)
    
    def inverse_kinematics(self, link: int, position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Compute the inverse kinematics and return the new joint values.

        Args:
            link (int): The link.
            position (x, y, z): Desired position of the link.
            orientation (x, y, z, w): Desired orientation of the link.

        Returns:
            List of joint values.
        """
        inverse_kinematics = self.sim.inverse_kinematics(self.body_name, link=link, position=position, orientation=orientation)
        return inverse_kinematics
    
    def position_control(self, target_q: list) -> None:
        """Position control the robot.

        Args:
            target_q (list): Target joint angles.
        """
        self.sim.position_control(self.body_name, target_q)

    def has_reached(self, target_position: np.ndarray, threshold: float) -> bool:
        joint_position = self.get_joint_angles_dof()
        return np.linalg.norm(joint_position - target_position) < threshold
    
    def get_base_orientation(self) -> np.ndarray:
        return self.sim.get_base_orientation(self.body_name)
    
    def get_base_position(self) -> np.ndarray:
        return self.sim.get_base_position(self.body_name)

class shadow(robot):
    def set_robot_info(self) -> None:
         self.body_name = "Shadow"
         self.urdf_path = osp.join(osp.dirname(__file__), '..', '..', '..', 'model', 'urdf', 'sr_description','urdf','shadow_hand_pybullet.urdf')
        #  self.base_position = position
        #  self.base_orientation = orentation
         #self.endeffect =[7, 12, 17, 23, 29]
         self.endeffect = [6, 11, 16, 22, 28]
         self.pre_grasp = np.array([0.0, 0.2, -0.2])
         self.init_joint = [0., 0., -0.349, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -0.349, 0., 0., 0., -1.047, 1.222, 0., 0., 0.]
         self.calibration_rot = rodrigues(np.array([0,-(np.pi/2),0]))[1] #mat
         self.calibration_trans = np.array([0.0000, -0.0100, 0.2130])
         
class allegro(robot):
    def set_robot_info(self) -> None:
        self.body_name = "Allegro"
        self.urdf_path = osp.join(osp.dirname(__file__), '..', '..', '..', 'model', 'urdf', 'allegro_hand','allegro_hand_description_right.urdf')
        self.endeffect = []
        self.pre_grasp = np.array([-0.2, 0.0, -0.2])

        self.init_joint = [0.0]*16
        self.calibration_rot = euler_to_mat(np.array([-torch.pi/2,torch.pi,torch.pi/2]), degrees=False) #mat
        self.calibration_trans = np.array([0.0500, 0.0000, 0.1000])

class barrett(robot):
    def set_robot_info(self) -> None:
        self.body_name = "Barrett"
        self.urdf_path = osp.join(osp.dirname(__file__), '..', '..', '..', 'model', 'urdf', 'barrett_adagrasp','model.urdf')
        self.endeffect = []
        self.pre_grasp = np.array([0.0, 0.0, -0.2]) 
        self.init_joint = [0.0]*8
        self.calibration_rot = euler_to_mat(np.array([0,torch.pi/2,torch.pi]), degrees=False) #mat
        self.calibration_trans = np.array([0.0000, 0.0000, 0.0000])

class robotiq(robot):
    def set_robot_info(self) -> None:
        self.body_name = "Robotiq"
        self.urdf_path = osp.join(osp.dirname(__file__), '..', '..', '..', 'model', 'urdf', 'robotiq_arg85','urdf','robotiq_arg85_description.urdf')
        # self.urdf_path = osp.join(osp.dirname(__file__), '..', '..', '..',  'model', 'urdf', 'robotiq_85v2', 'robotiq_2f_85_v3.urdf')
       
        self.endeffect = []
        self.pre_grasp = np.array([0.0, 0.0, -0.2])
        self.init_joint = [0.0]*6
        self.calibration_rot = euler_to_mat(np.array([torch.pi/2,torch.pi,3*torch.pi/2]), degrees=False) #mat
        self.calibration_trans = np.array([0.0000, 0.0000, 0.0000])

class MountingBase():
    def __init__(self, 
                 sim: PybulletBase,
                 gripper: robot,
                 ) -> None:
        self.sim = sim
        self.body_name = "base"
        self.urdf_path = osp.join(osp.dirname(__file__), 'xyz.urdf')
        self.gripper = gripper
        
    def mounting_gripper(self):
        m_pose, m_quat = self.sim.get_base_position(self.gripper.body_name), self.sim.get_base_orientation(self.gripper.body_name)
        self.sim.loadURDF(body_name=self.body_name, 
                                  fileName=self.urdf_path, 
                                  basePosition=m_pose, 
                                  baseOrientation=m_quat, 
                                  useFixedBase=True)
        
        for joint in range(self.sim.get_num_joints(self.body_name)):
            info = self.sim.pc.getJointInfo(self.sim._bodies_idx[self.body_name], joint)
            if info[12].decode('utf-8') == "end_effector_link":
                eelink_id = info[0]
        
        c_id = self.sim.pc.createConstraint(self.sim._bodies_idx[self.body_name], eelink_id, self.sim._bodies_idx[self.gripper.body_name], -1, 
                                            jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                            parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0],
                                            parentFrameOrientation=[0, 0, 0, 1], childFrameOrientation=[0, 0, 0, 1]
        )
        
        # self.sim.pc.setJointMotorControlArray(self.sim._bodies_idx[self.body_name], 
        #                                       list(range(self.sim.get_num_joints(self.body_name))), 
        #                                       controlMode=p.POSITION_CONTROL, 
        #                                       targetPositions=[0.0]*self.sim.get_num_joints(self.body_name),
        #                                       forces=[1000.0]*self.sim.get_num_joints(self.body_name))
        return self.sim._bodies_idx[self.body_name], c_id

    def reset_mount(self, position: np.ndarray, orientation: np.ndarray):
        self.sim.set_base_pose(self.body_name, position, orientation)
    
    def get_num_joints(self):
        return self.sim.get_num_joints(self.body_name)
    
    def get_base_position(self):
        return self.sim.get_base_position(self.body_name)
    
    def get_base_orientation(self):
        return self.sim.get_base_orientation(self.body_name)
    
    def get_joint_angles(self):
        return self.sim.get_joint_angles(self.body_name, list(range(self.get_num_joints())))