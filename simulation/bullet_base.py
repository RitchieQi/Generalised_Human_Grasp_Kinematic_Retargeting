import os
osp = os.path
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc
import numpy as np
from typing import List, Optional, Tuple, Union, Iterator, Any, Dict
from dex_ycb_toolkit.dex_ycb import DexYCBDataset
from scipy.spatial.transform import Rotation as R
import trimesh as tm
import cv2
import os

class PybulletBase:
    def __init__(
        self,
        connection_mode: str = "GUI",
    ):
        self.connect = p.DIRECT if connection_mode == "DIRECT" else p.GUI
        # self.pc = bc.BulletClient(connection_mode=self.connect)
        # self.pc.setAdditionalSearchPath(pd.getDataPath())
        # # self.pc.loadURDF("plane.urdf", [0, 0, -0.5], useFixedBase=True)
        # self.pc.setTimeStep(1 / 1000)
        # self.pc.setGravity(0, 0, -9.8)
        # self.pc.setPhysicsEngineParameter(numSolverIterations=200)  # Increase solver accuracy
        # self.pc.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])
        # self.pc.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # # self.pc.setGravity(0, 0, 0) #set gravity to 0 as the optimization is done with the net force = 0 not the net force = gravity
        # self._bodies_idx = {}
        # self.pc.setRealTimeSimulation(False)
        # self.n_steps = 0
        # self.lowest_point = None
        self.control_joint: list[float] = []
        
    
    def _connect_(self) -> None:
        """Connect to the simulation"""
        self.pc = bc.BulletClient(connection_mode=self.connect)
        self.pc.setAdditionalSearchPath(pd.getDataPath())
        # self.pc.loadURDF("plane.urdf", [0, 0, -0.5], useFixedBase=True)
        self.pc.setTimeStep(1 / 1000)
        self.pc.setGravity(0, 0, -9.8)
        self.pc.setPhysicsEngineParameter(numSolverIterations=200)  # Increase solver accuracy
        
        
        camera_distance = 0.6 # Distance from camera to target
        camera_yaw = 0       # Horizontal angle (degrees)
        camera_pitch = -30     # Vertical angle (degrees)
        camera_target = [0, 0, 0]
        
        
        self.pc.resetDebugVisualizerCamera(cameraDistance=camera_distance, cameraYaw=camera_yaw, cameraPitch=camera_pitch, cameraTargetPosition=camera_target)
        self.pc.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        
        self.width, self.height = 1920, 1080

        self.view_matrix = self.pc.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=camera_target,
                    distance=camera_distance,
                    yaw=camera_yaw,
                    pitch=camera_pitch,
                    roll=0,
                    upAxisIndex=2  # Z-axis is up (2 for Z, 1 for Y)
                    )

        fov = 60  # Degrees
        aspect = self.width / self.height
        near, far = 0.1, 100.0
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=near,
            farVal=far
)

        
        # self.pc.setGravity(0, 0, 0) #set gravity to 0 as the optimization is done with the net force = 0 not the net force = gravity
        self._bodies_idx = {}
        self.pc.setRealTimeSimulation(False)
        self.n_steps = 0
        self.lowest_point = None
        self.control_joint = None
        

    # @property
    # def dt(self):
    #     return self.timeStep * self.n_steps
    
    def step(self) -> None:
        self.pc.stepSimulation()
    
    def close(self) -> None:
        """Close the simulation"""
        if self.pc.isConnected():
            self.pc.disconnect()
    
    def get_base_position(self, body: str) -> np.ndarray:
        """Get the position of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The position, as (x, y, z).
        """
        position = self.pc.getBasePositionAndOrientation(self._bodies_idx[body])[0]
        return np.array(position)
    
    def get_base_orientation(self, body: str) -> np.ndarray:
        """Get the orientation of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The orientation, as (x, y, z, w).
        """
        orientation = self.pc.getBasePositionAndOrientation(self._bodies_idx[body])[1]
        return np.array(orientation)
    
    def get_base_rotation(self, body: str, type: str = "euler") -> np.ndarray:
        """Get the rotation of the body.

        Args:
            body (str): Body unique name.
            type (str): Type of angle, either "euler" or "quaternion"

        Returns:
            np.ndarray: The rotation.
        """
        quaternion = self.get_base_orientation(body)
        if type == "euler":
            rotation = self.pc.getEulerFromQuaternion(quaternion)
            return np.array(rotation)
        elif type == "quaternion":
            return np.array(quaternion)
        else:
            raise ValueError("""type must be "euler" or "quaternion".""")
   
    def get_base_velocity(self, body: str) -> np.ndarray:
        """Get the velocity of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The velocity, as (vx, vy, vz).
        """
        velocity = self.pc.getBaseVelocity(self._bodies_idx[body])[0]
        return np.array(velocity)

    def get_base_angular_velocity(self, body: str) -> np.ndarray:
        """Get the angular velocity of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The angular velocity, as (wx, wy, wz).
        """
        angular_velocity = self.pc.getBaseVelocity(self._bodies_idx[body])[1]
        return np.array(angular_velocity)
    
    def get_link_position(self, body: str, link: int) -> np.ndarray:
        """Get the position of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The position, as (x, y, z).
        """
        position = self.pc.getLinkState(self._bodies_idx[body], link)[0]
        return np.array(position)

    def get_link_orientation(self, body: str, link: int) -> np.ndarray:
        """Get the orientation of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The rotation, as (rx, ry, rz).
        """
        orientation = self.pc.getLinkState(self._bodies_idx[body], link)[1]
        return np.array(orientation)

    def get_link_velocity(self, body: str, link: int) -> np.ndarray:
        """Get the velocity of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The velocity, as (vx, vy, vz).
        """
        velocity = self.pc.getLinkState(self._bodies_idx[body], link, computeLinkVelocity=True)[6]
        return np.array(velocity)

    def get_link_angular_velocity(self, body: str, link: int) -> np.ndarray:
        """Get the angular velocity of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The angular velocity, as (wx, wy, wz).
        """
        angular_velocity = self.pc.getLinkState(self._bodies_idx[body], link, computeLinkVelocity=True)[7]
        return np.array(angular_velocity)       

    def get_joint_angle(self, body: str, joint: int) -> float:
        """Get the angle of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body

        Returns:
            float: The angle.
        """
        return self.pc.getJointState(self._bodies_idx[body], joint)[0]
    
    def get_joint_angles(self, body: str, joints: list) -> np.ndarray:
        """Get the angles of the joints of the body.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.

        Returns:
            np.ndarray: The angles.
        """
        return np.array([self.get_joint_angle(body, j) for j in joints])
    
    def get_joint_velocity(self, body: str, joint: int) -> float:
        """Get the velocity of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body

        Returns:
            float: The velocity.
        """
        return self.pc.getJointState(self._bodies_idx[body], joint)[1]
    
    def get_joint_velocities(self, body: str, joints: list) -> np.ndarray:
        """Get the velocities of the joints of the body.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.

        Returns:
            np.ndarray: The velocities.
        """
        return np.array([self.get_joint_velocity(body, j) for j in joints])

    def set_base_pose(self, body: str, position: np.ndarray, orientation: np.ndarray) -> None:
        """Set the position of the body.

        Args:
            body (str): Body unique name.
            position (np.ndarray): The position, as (x, y, z).
            orientation (np.ndarray): The target orientation as quaternion (x, y, z, w).
        """
        if len(orientation) == 3:
            orientation = self.pc.getQuaternionFromEuler(orientation)
        self.pc.resetBasePositionAndOrientation(
            bodyUniqueId=self._bodies_idx[body], posObj=position, ornObj=orientation
        )

    def set_joint_angles(self, body: str, joints: list, angles: list) -> None:
        """Set the angles of the joints of the body.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.
            angles (np.ndarray): List of target angles, as a list of floats.
        """
        for joint, angle in zip(joints, angles):
            self.set_joint_angle(body=body, joint=joint, angle=angle)
    
    def set_joint_angles_dof(self, body: str, angles: list) -> None:
        """Set the angles of the joints of the body.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.
            angles (np.ndarray): List of target angles, as a list of floats.
        """
        if self.control_joint is None:
            self.get_controllable_joints(body)
        assert len(angles) == len(self.control_joint), "The number of angles must match the number of controllable joints."
        
        for joint, angle in zip(self.control_joint, angles):
            self.set_joint_angle(body=body, joint=joint, angle=angle)
    
    def set_joint_angle(self, body: str, joint: int, angle: float) -> None:
        """Set the angle of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body.
            angle (float): Target angle.
        """
        self.pc.resetJointState(bodyUniqueId=self._bodies_idx[body], jointIndex=joint, targetValue=angle)
    
    def inverse_kinematics(self, body: str, link: int, position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Compute the inverse kinematics and return the new joint state.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            position (np.ndarray): Desired position of the end-effector, as (x, y, z).
            orientation (np.ndarray): Desired orientation of the end-effector as quaternion (x, y, z, w).

        Returns:
            np.ndarray: The new joint state.
        """
        joint_state = self.pc.calculateInverseKinematics(
            bodyIndex=self._bodies_idx[body],
            endEffectorLinkIndex=link,
            targetPosition=position,
            targetOrientation=orientation,
        )
        return np.array(joint_state)

    def get_num_joints(self, body: str) -> int:
        """Get the number of joints of the body.

        Args:
            body (str): Body unique name.

        Returns:
            int: The number of joints.
        """
        return self.pc.getNumJoints(self._bodies_idx[body])

    def get_controllable_joints(self, body: str) -> None:
        """Get the indices of the controllable joints of the body.

        Args:
            body (str): Body unique name.

        Returns:
            list: List of joint indices.
        """
        joints_num = self.get_num_joints(body)
        self.control_joint = []
        for i in range(joints_num):
            joint_info = self.pc.getJointInfo(self._bodies_idx[body], i)
            if joint_info[2] == 0:
                self.control_joint.append(i)
        
    def get_joint_angles_dof(self, body: str) -> np.ndarray:
        """Get the degree of freedom of the joints of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The degree of freedom.
        """
        if not self.control_joint:
            self.get_controllable_joints(body)

        joint_angle_dof = self.get_joint_angles(body, self.control_joint)
        return joint_angle_dof

    def get_joint_velocities_dof(self, body: str) -> np.ndarray:
        """Get the velocities of the joints of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The velocities.
        """
        if not self.control_joint:
            self.get_controllable_joints(body)

        joint_velocity_dof = self.get_joint_velocities(body, self.control_joint)
        return joint_velocity_dof
    

    def get_dynamics_matrices(self, body: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the mass matrix, h matrix (which is sum of coriolis and gravity), and gravity vector.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The mass matrix, h matrix, and gravity vector.
        """
        
        joint_positions = self.get_joint_angles_dof(body)
        joint_velocities = self.get_joint_velocities_dof(body)
        zero_vec = [0.0] * len(joint_positions)
        
        M = self.pc.calculateMassMatrix(self._bodies_idx[body], joint_positions)
        G = self.pc.calculateInverseDynamics(self._bodies_idx[body], joint_positions, joint_velocities, zero_vec)
        
        return M, G
    
    def loadURDF(self, body_name: str, **kwargs: Any) -> None:
        """Load URDF file.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
        """
        self._bodies_idx[body_name] = self.pc.loadURDF(**kwargs)


        
    def load_obj_as_mesh(self, body_name: str, obj_path: str, position: np.ndarray, orientation: np.ndarray, obj_mass: float=0.0, frictional_coe: float = 0.0) -> None:
        """Load obj file and create mesh/collision shape from it.
            ref: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/createVisualShape.py
        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            obj_path (str): Path to the obj file.
        """
        """
        Convert the mesh to vhacd
        """
        
        obj_fn = obj_path.split(os.sep)[-1]
        obj_fn_no_ext = obj_fn.split('.')[0]
        vhacd_path = os.path.join(os.path.dirname(obj_path), obj_fn_no_ext + '_vhacd.obj')
        vhacd_log_path = os.path.join(os.path.dirname(obj_path), obj_fn_no_ext + '_vhacd.log')
        
        if not os.path.exists(vhacd_path):
            p.vhacd(fileNameIn=obj_path, fileNameOut=vhacd_path, fileNameLogging=vhacd_log_path, resolution=50000)
        
        obj_visual_shape_id = self.pc.createVisualShape(
                                                    shapeType=p.GEOM_MESH, 
                                                    fileName=obj_path, 
                                                    rgbaColor=[1, 1, 1, 1],
                                                    meshScale=[1, 1, 1])

        obj_collision_shape_id = self.pc.createCollisionShape(
                                                shapeType = p.GEOM_MESH,
                                                fileName = vhacd_path,
                                                # flags=p.GEOM_FORCE_CONCAVE_TRIMESH |
                                                # p.GEOM_CONCAVE_INTERNAL_EDGE,
                                                # flags=p.GEOM_FORCE_CONVEX_MESH,
                                                meshScale=[1, 1, 1],
                                                )
        
        self._bodies_idx[body_name] = self.pc.createMultiBody(
                                baseMass=obj_mass,
                                baseInertialFramePosition=[0, 0, 0],
                                baseCollisionShapeIndex=obj_collision_shape_id,
                                baseVisualShapeIndex=obj_visual_shape_id,
                                basePosition=position,
                                baseOrientation=orientation,
                                useMaximalCoordinates=True)
        
        self.pc.changeDynamics(self._bodies_idx[body_name], -1, 
                                lateralFriction=frictional_coe, 
                                restitution=0.0,
                                spinningFriction=0.01,
                                rollingFriction=0.01,
                                mass = obj_mass,
                                contactStiffness=5000,
                                contactDamping=200,
                                collisionMargin=0.001,
                                ccdSweptSphereRadius=0.001,
                                contactProcessingThreshold=0.001,
                                )
        
        mesh = tm.load(obj_path, force='mesh', process=False)
        mesh.apply_transform(np.vstack((np.hstack((R.from_quat(orientation).as_matrix(), position.reshape(3,1))), np.array([0, 0, 0, 1]))))
        verts = np.array(mesh.vertices)
        lowest_point_z = np.min(verts[:,2])
        self.lowest_point = lowest_point_z
        
        
        # visual_data = self.pc.getVisualShapeData(self._bodies_idx[body_name])
        
    def load_plane(self) -> None:
        if self.lowest_point is None:
            raise ValueError("Please load an object first.")
        self._bodies_idx["plane"] = self.pc.loadURDF("plane.urdf", [0, 0, self.lowest_point-0.5], useFixedBase=True)
        self.pc.changeDynamics(self._bodies_idx["plane"], -1, lateralFriction=1.0, restitution=0.0)
        
    def set_lateral_friction(self, body: str, link: list, lateral_friction: float) -> None:
        """Set the lateral friction of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            lateral_friction (float): Lateral friction.
        """
        for lk in link:
            self.pc.changeDynamics(
                bodyUniqueId=self._bodies_idx[body],
                linkIndex=lk,
                lateralFriction=lateral_friction,
            ) 
        
    def set_mass(self, body: str, link: int, mass: float) -> None:
        """Set the lateral friction of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            lateral_friction (float): Lateral friction.
        """
        self.pc.changeDynamics(
            bodyUniqueId=self._bodies_idx[body],
            linkIndex=link,
            mass=mass,
        ) 
        
    def set_spinning_friction(self, body: str, link: int, spinning_friction: float) -> None:
        """Set the spinning friction of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            spinning_friction (float): Spinning friction.
        """
        self.pc.changeDynamics(
            bodyUniqueId=self._bodies_idx[body],
            linkIndex=link,
            spinningFriction=spinning_friction,
        )

    def set_rolling_friction(self, body: str, link: int, rolling_friction: float) -> None:
        """Set the rolling friction of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            rolling_friction (float): Rolling friction.
        """
        self.pc.changeDynamics(
            bodyUniqueId=self._bodies_idx[body],
            linkIndex=link,
            rollingFriction=rolling_friction,
        )
        
    def set_friction_all(self, body: str, lateral_friction: float) -> None:
        """Set the lateral friction of all links.

        Args:
            body (str): Body unique name.
            lateral_friction (float): Lateral friction.
        """
        for i in range(-1, self.get_num_joints(body)):
            self.pc.changeDynamics(
                bodyUniqueId=self._bodies_idx[body],
                linkIndex=i,
                lateralFriction=lateral_friction,
                spinningFriction=0.01,
                rollingFriction=0.01,
            )
            
    def remove_body(self, body: str) -> None:
        """Remove the body.

        Args:
            body (str): Body unique name.
        """
        self.pc.removeBody(self._bodies_idx[body])

    def position_control(self, body: str, traget_q: list) -> None:
        """Position control the robot.

        Args:
            body (str): Body unique name.
            traget_q (list): Target joint angles.
        """
        if not self.control_joint:
            self.get_controllable_joints(body)
        assert len(traget_q) == len(self.control_joint), "The number of angles must match the number of controllable joints."
        
        for i in range(len(self.control_joint)):
            self.pc.setJointMotorControl2(
                bodyIndex=self._bodies_idx[body],
                jointIndex=self.control_joint[i],
                controlMode=p.POSITION_CONTROL,
                targetPosition=traget_q[i],
            )
    
    def disable_motor(self, body: str) -> None:
        if not self.control_joint:
            self.get_controllable_joints(body)
        for i in self.control_joint:
            self.pc.setJointMotorControl2(
                bodyIndex=self._bodies_idx[body],
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0,
                force=0,
            )

    def setup_dynamics(self, body: str) -> None:
        self.pc.changeDynamics(self._bodies_idx[body], -1, 
                                   mass=0.1,
                                   restitution=0.0,
                                   )
        for i in range(0, self.get_num_joints(body)):
            self.pc.changeDynamics(
                bodyUniqueId=self._bodies_idx[body],
                linkIndex=i,
                restitution=0.0,
                contactStiffness=5000,
                contactDamping=200,
                collisionMargin=0.001,
                ccdSweptSphereRadius=0.001,
                contactProcessingThreshold=0.001,
                )
                
    def start_record(self, file_name: str) -> None:
        self.pc.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, file_name)
    
    def stop_record(self) -> None:
        self.pc.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)
        
    def reset_simulation(self) -> None:
        self.pc.resetSimulation()
        for constraint_id in range(self.pc.getNumConstraints()):
            self.pc.removeConstraint(constraint_id)

        # for body_id in range(self.pc.getNumBodies()):
        #     self.pc.applyExternalForce(body_id, -1, [0, 0, 0], [0, 0, 0], self.pc.WORLD_FRAME)
            
    def remove_all(self) -> None:
        objects = [self.pc.getBodyUniqueId(i) for i in range(self.pc.getNumBodies())]
        
        for obj in objects:
            self.pc.removeBody(obj)
    
    def get_camera_image(self) -> np.ndarray:

        
        width_m, height_m, rgb, depth, seg = self.pc.getCameraImage(self.width, 
                                                                    self.height,
                                                                    self.view_matrix,
                                                                    self.proj_matrix,
                                                                    renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb = np.array(rgb, dtype=np.uint8)
        rgb_array = rgb.reshape((self.height, self.width, 4))[:, :, :3]  # Remove alpha
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)  
        return bgr_array
    
    def reset_jointstate(self, body: str, joint: np.ndarray ) -> None:
        for i in range(self.get_num_joints(body)):
            self.pc.resetJointState(self._bodies_idx[body], i, joint[i])