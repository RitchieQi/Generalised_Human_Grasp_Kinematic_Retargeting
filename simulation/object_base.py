
import os
osp = os.path
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc
import numpy as np
from typing import List, Optional, Tuple, Union, Iterator, Any, Dict
from scipy.spatial.transform import Rotation as R
import trimesh as tm
import os
from bullet_base import PybulletBase
class objects:
    def __init__(
        self,
        sim: PybulletBase,
    )-> None:
        self.sim = sim
        self.obj_path = None
        self.position = None
        self.orientation = None
        self.obj_mass = None
        self.body_name = "object"
        self.mesh_cache = None
        self.vhacd_path = None
        self.global_constraint = None
    def convert_vhacd(self) -> None:
        """
        Convert the mesh to vhacd
        """
        obj_fn = self.obj_path.split(os.sep)[-1]
        obj_fn_no_ext = obj_fn.split('.')[0]
        vhacd_path = os.path.join(os.path.dirname(self.obj_path), obj_fn_no_ext + '_vhacd.obj')
        vhacd_log_path = os.path.join(os.path.dirname(self.obj_path), obj_fn_no_ext + '_vhacd.log')
        
        if not os.path.exists(vhacd_path):
            p.vhacd(fileNameIn=self.obj_path, fileNameOut=vhacd_path, fileNameLogging=vhacd_log_path, resolution=50000)
        
        self.vhacd_path = vhacd_path
    
    
    def load_vhacd_cache(self) -> None:
        """
        load the mesh as trimesh object
        """    
        self.mesh_cache = tm.load(self.vhacd_path, force='mesh', process=False)
    
    def set_object_info(self, file_path: str, position: np.ndarray, orientation: np.ndarray, obj_mass: float) -> None:
        self.obj_path = file_path
        self.position = position
        self.orientation = orientation
        self.obj_mass = obj_mass
    
    def load_object(self, mu) -> None:
        self.sim.load_obj_as_mesh(body_name=self.body_name, 
                                    obj_path=self.obj_path, 
                                    position=self.position, 
                                    orientation=self.orientation,
                                    obj_mass=self.obj_mass,
                                    frictional_coe=mu)
    
    def remove_object(self) -> None:
        self.sim.remove_body("object")  
        
    def set_mass(self, mass: float) -> None:
        self.sim.set_mass("object", link=-1, mass=mass)
        
    # def stable_position(self) -> None:
    def mesh_inertia(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.mesh_cache:
            self.load_vhacd_cache()
        self.mesh_cache.density = self.obj_mass / self.mesh_cache.volume
        return self.mesh_cache.moment_inertia, self.mesh_cache.center_mass

    def centroid(self) -> np.ndarray:
        if not self.mesh_cache:
            self.load_vhacd_cache()
        return self.mesh_cache.centroid
    
    def center_of_mass(self) -> np.ndarray:
        if not self.mesh_cache:
            self.load_vhacd_cache()
        return self.mesh_cache.center_mass
    
    def create_world_contraint(self) -> None:
        fix_base = self.sim.pc.createMultiBody(
            baseMass=0,  # Mass = 0 to keep it static
            baseCollisionShapeIndex=-1,  # No collision shape (invisible)
            baseVisualShapeIndex=-1,  # No visual shape
            basePosition=self.position,  # Same position as the object
            baseOrientation=self.orientation  # Same orientation as the object
            )
        
        self.global_constraint = self.sim.pc.createConstraint(
            parentBodyUniqueId=fix_base,
            parentLinkIndex=-1,
            childBodyUniqueId=self.sim._bodies_idx[self.body_name],
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
    
    def create_mount_contraint(self) -> None:
        m_pose, m_quat = self.sim.get_base_position(self.body_name), self.sim.get_base_orientation(self.body_name)
        self.sim.loadURDF(  body_name="obj_mount",
                            fileName=osp.join(osp.dirname(__file__), 'xyz.urdf'),
                            basePosition=m_pose,
                            baseOrientation=m_quat,
                            useFixedBase=True)
        for joint in range(self.sim.get_num_joints("obj_mount")):
            info = self.sim.pc.getJointInfo(self.sim._bodies_idx["obj_mount"], joint)
            if info[12].decode('utf-8') == "end_effector_link":
                eelink_id = info[0]
                
        self.global_constraint = self.sim.pc.createConstraint(
            parentBodyUniqueId=self.sim._bodies_idx["obj_mount"],
            parentLinkIndex=eelink_id,
            childBodyUniqueId=self.sim._bodies_idx[self.body_name],
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        
        for joint in range(self.sim.get_num_joints("obj_mount")):
            self.sim.pc.setJointMotorControl2(
                bodyIndex=self.sim._bodies_idx["obj_mount"], 
                jointIndex=joint,
                controlMode=p.POSITION_CONTROL, 
                targetPosition=0,
                positionGain=0.2,
                velocityGain=0.1,
                force=10.0)
        
    def remove_world_contraint(self) -> None:
        self.sim.pc.removeConstraint(self.global_constraint)
    
    def generate_urdf(self) -> None:
        # Generate URDF file
        name = self.body_name
        mass = self.obj_mass
        origin = [0,0,0]