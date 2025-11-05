import os
osp = os.path
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc
import numpy as np
from typing import List, Optional, Tuple, Union, Iterator, Any, Dict
import os


class Trajectory:
    def __init__(self, initial_positions: np.ndarray, initial_joint:np.ndarray, grasp_position: np.ndarray, grasp_joint: np.ndarray, lift_up: np.ndarray, time_step = 1000) -> None:
        self.initial_positions = initial_positions
        self.initial_joint = initial_joint
        self.grasp_position = grasp_position
        self.grasp_joint = grasp_joint
        self.liftup_position = lift_up
        self.tm = time_step
        self.pre_grasp = self.grasp_position*0.5
        self.time_interval = [0, 2, 2, 3]
        # print("pre_grasp", self.pre_grasp)
        # print("liftup", self.liftup_position)
        # self.coefficients = None        
    
    @property
    def get_duration(self):
        return self.time_interval[-1]
    
    @property
    def get_reach_duration(self):
        return self.time_interval[-2]
    
    def cubic_motion_planning(self, initial_positions: np.ndarray, target_positions: np.ndarray, t0: int, tf: int, num_points: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Plan a cubic motion trajectory for multiple joints.
        
        initial_positions: List of initial joint positions
        target_positions: List of target joint positions
        initial_velocities: List of initial joint velocities
        target_velocities: List of target joint velocities
        t0: Initial time
        tf: Final time
        num_points: Number of points in the trajectory
        
        Returns:
        positions: Joint positions trajectory
        velocities: Joint velocities trajectory
        accelerations: Joint accelerations trajectory
        time_steps: Time steps of the trajectory
        """
        initial_velocities = np.zeros_like(initial_positions) 
        target_velocities = np.zeros_like(target_positions)
        
        def cubic_spline_coefficients(q0, qf, v0, vf, t0, tf):
            A = np.array([[1, t0, t0**2, t0**3],
                        [0, 1, 2*t0, 3*t0**2],
                        [1, tf, tf**2, tf**3],
                        [0, 1, 2*tf, 3*tf**2]])
            b = np.array([q0, v0, qf, vf])
            coefficients = np.linalg.solve(A, b)
            return coefficients

        def cubic_spline_trajectory(coefficients, t):
            a0, a1, a2, a3 = coefficients
            q = a0 + a1*t + a2*t**2 + a3*t**3
            q_dot = a1 + 2*a2*t + 3*a3*t**2
            q_ddot = 2*a2 + 6*a3*t
            return q, q_dot, q_ddot

        num_joints = len(initial_positions)
        time_steps = np.linspace(t0, tf, num_points)

        positions = np.zeros((num_points, num_joints))
        velocities = np.zeros((num_points, num_joints))
        accelerations = np.zeros((num_points, num_joints))

        for i in range(num_joints):
            coeffs = cubic_spline_coefficients(initial_positions[i], target_positions[i],
                                            initial_velocities[i], target_velocities[i],
                                            t0, tf)
            for j, t in enumerate(time_steps):
                q, q_dot, q_ddot = cubic_spline_trajectory(coeffs, t)
                positions[j, i] = q
                velocities[j, i] = q_dot
                accelerations[j, i] = q_ddot

        return positions, velocities, accelerations, time_steps
    
    def solve_cube_polynomial(self, coefficients, x, t_interval):
        """
        Solve the cubic polynomial for t given x.
        a3*t^3 + a2*t^2 + a1*t + (a0 - x) = 0
        """
        
        a0, a1, a2, a3 = coefficients
        new_coe = [a3, a2, a1, a0 - x]
        roots = np.roots(new_coe)
        real_roots = [t.real for t in roots if np.isreal(t) and t_interval[0] <= t.real <= t_interval[1]]
        if real_roots:
            return min(real_roots)
        else:
            return None
        
    def cubic_spline_coefficients(self, q0, qf, v0, vf, t0, tf):
            A = np.array([[1, t0, t0**2, t0**3],
                        [0, 1, 2*t0, 3*t0**2],
                        [1, tf, tf**2, tf**3],
                        [0, 1, 2*tf, 3*tf**2]])
            b = np.array([q0, v0, qf, vf])
            coefficients = np.linalg.solve(A, b)
            return coefficients            
    
    def cubic_spline_trajectory(self, coefficients, t):
            a0, a1, a2, a3 = coefficients
            q = a0 + a1*t + a2*t**2 + a3*t**3
            q_dot = a1 + 2*a2*t + 3*a3*t**2
            q_ddot = 2*a2 + 6*a3*t
            return q, q_dot, q_ddot
    
    def basic_motion_planning(self, initial_positions: np.ndarray, target_positions: np.ndarray, t0: int, tf: int, num_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        initial_velocities = np.zeros_like(initial_positions)
        target_velocities = np.zeros_like(target_positions)
        num_joints = len(initial_positions) 
        time_steps = np.linspace(t0, tf, num_points)
        
        positions = np.zeros((num_points, num_joints))
        velocities = np.zeros((num_points, num_joints))
        accelerations = np.zeros((num_points, num_joints))
        
        for i in range(num_joints):
            coeffs = self.cubic_spline_coefficients(initial_positions[i], target_positions[i],
                                            initial_velocities[i], target_velocities[i],
                                            t0, tf)
            for j, t in enumerate(time_steps):
                q, q_dot, q_ddot = self.cubic_spline_trajectory(coeffs, t)
                positions[j, i] = q
                velocities[j, i] = q_dot
                accelerations[j, i] = q_ddot

        return positions, velocities, accelerations, time_steps        
    
    def basic_motion_planning_asynb(self, initial_positions: np.ndarray, target_positions: np.ndarray, via_position: np.ndarray, t0: int, tf: int, num_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        initial_velocities = np.zeros_like(initial_positions)
        target_velocities = np.zeros_like(target_positions)
        
        num_joints = len(initial_positions) 
        time_steps = np.linspace(t0, tf, num_points)
        # print("chect base setup", initial_positions, target_positions, via_position, t0, tf)
        positions = np.zeros((num_points, num_joints))
        velocities = np.zeros((num_points, num_joints))
        accelerations = np.zeros((num_points, num_joints))
        
        via_time = np.zeros_like(target_positions)
        
        for i in range(num_joints):
            coeffs = self.cubic_spline_coefficients(initial_positions[i], target_positions[i],
                                            initial_velocities[i], target_velocities[i],
                                            t0, tf)
            via_time[i] = self.solve_cube_polynomial(coeffs, via_position[i], [t0, tf])
            for j, t in enumerate(time_steps):
                q, q_dot, q_ddot = self.cubic_spline_trajectory(coeffs, t)
                positions[j, i] = q
                velocities[j, i] = q_dot
                accelerations[j, i] = q_ddot
        timestamp = (np.nanmean(via_time) * self.tm).astype(int)
        return positions, velocities, accelerations, time_steps, timestamp, np.nanmean(via_time)
        
    def basic_motion_planning_asynj(self, initial_positions: np.ndarray, target_positions: np.ndarray, tf: int, tt: int, timestamp: int, num_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        initial_velocities = np.zeros_like(initial_positions)
        target_velocities = np.zeros_like(target_positions)
        
        num_joints = len(initial_positions)
        time_steps = np.linspace(tf, tt, num_points-timestamp)
        
        positions = np.zeros((num_points-timestamp, num_joints))
        velocities = np.zeros((num_points-timestamp, num_joints))
        accelerations = np.zeros((num_points-timestamp, num_joints))
        
        for i in range(num_joints):
            coeffs = self.cubic_spline_coefficients(initial_positions[i], target_positions[i],
                                            initial_velocities[i], target_velocities[i],
                                            tf, tt)
            for j, t in enumerate(time_steps):
                q, q_dot, q_ddot = self.cubic_spline_trajectory(coeffs, t)
                positions[j, i] = q
                velocities[j, i] = q_dot
                accelerations[j, i] = q_ddot
        
        return positions, velocities, accelerations, time_steps
        
        
    def cubic_motion_planning_syn(self, initial_positions: Tuple[np.ndarray,np.ndarray], target_positions: Tuple[np.ndarray,np.ndarray], t0: int, tf: int, num_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,np.ndarray, np.ndarray, np.ndarray, np.ndarray]:   
        base_init_pos, gripper_init_pos = initial_positions
        base_target_pos, gripper_target_pos = target_positions
        qb, vb, ab, tb = self.basic_motion_planning(base_init_pos, base_target_pos, t0, tf, num_points)
        qg, vg, ag, tg = self.basic_motion_planning(gripper_init_pos, gripper_target_pos, t0, tf, num_points)
        return qb, vb, ab, tb, qg, vg, ag, tg
        
    def cubic_motion_planning_asyn(self, initial_positions: Tuple[np.ndarray,np.ndarray], target_positions: Tuple[np.ndarray,np.ndarray], via_position: np.ndarray, t0: int, tf: int, tt: int, num_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Moving the base earlier than the gripper
        Base:  -------------->__________
        Joint: ___________------------->
        """
        base_init_pos, gripper_init_pos = initial_positions
        base_target_pos, gripper_target_pos = target_positions
        num_base = len(base_init_pos)
        num_gripper = len(gripper_init_pos)
        qb_ = np.ones((num_points, num_base))*base_target_pos
        qj_ = np.ones((num_points, num_gripper))*gripper_init_pos
        
        vb_ = np.zeros((num_points, num_base))
        vj_ = np.zeros((num_points, num_gripper))
        
        ab_ = np.zeros((num_points, num_base))
        aj_ = np.zeros((num_points, num_gripper))
        
        # tb_ = np.linspace(t0, tf, num_points)
        # tj_ = np.linspace(t0, tt, num_points)
        
        
        qb, vb, ab, tb, timestamp, tv = self.basic_motion_planning_asynb(base_init_pos, base_target_pos, via_position, t0, tf, int(self.tm*(tf-t0)))
        # print("check joint setup", len(gripper_init_pos), len(gripper_target_pos), tv, tt, timestamp, num_points)
        qj, vj, aj, tj = self.basic_motion_planning_asynj(gripper_init_pos, gripper_target_pos, tv, tt, timestamp, num_points)    
        
        
        
        qb_[:int(self.tm*(tf-t0))], vb_[:int(self.tm*(tf-t0))], ab_[:int(self.tm*(tf-t0))] = qb, vb, ab
        
        qj_[timestamp:], vj_[timestamp:], aj_[timestamp:] = qj, vj, aj
        
        return qb_, vb_, ab_, tb, qj_, vj_, aj_, tj    
    
    def trajectory_generate_v2(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        time_itv = self.time_interval
        qb_0, vb_0, ab_0, tb_0, qj_0, vj_0, aj_0, tj_0 = self.cubic_motion_planning_asyn((self.initial_positions, self.initial_joint), (self.grasp_position, self.grasp_joint), self.pre_grasp, time_itv[0], time_itv[1], time_itv[2], int(self.tm*(time_itv[2]-time_itv[0])))
        qb_1, vb_1, ab_1, tb_1, qj_1, vj_1, aj_1, tj_1 = self.cubic_motion_planning_syn((self.grasp_position, self.grasp_joint), (self.liftup_position, self.grasp_joint), time_itv[2], time_itv[3], int(self.tm*(time_itv[3]-time_itv[2])))
        # print("check size", qb_0.shape, qb_1.shape, qj_0.shape, qj_1.shape, tb_0.shape, tb_1.shape, tj_0.shape, tj_1.shape, ab_0.shape, ab_1.shape, aj_0.shape, aj_1.shape)
        
        return np.concatenate((qb_0, qb_1)), np.concatenate((vb_0, vb_1)), np.concatenate((ab_0, ab_1)), np.concatenate((tb_0, tb_1)), np.concatenate((qj_0, qj_1)), np.concatenate((vj_0, vj_1)), np.concatenate((aj_0, aj_1)), np.concatenate((tj_0, tj_1))
    
    def trajectory_generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        from the pre_grasp to the grasp position:
            base: move from the pre_grasp to the grasp position
            gripper: stay
            
        grasp position:
            base: stay
            gripper: move from the initial joint to the grasp joint
            
        from the grasp position to the lift up position:
            base: move from the grasp position to the lift up position
            gripper: stay
            
            
        need add on more pre-grasp position    
        """
        time_interval = self.time_interval
        qb_0, vb_0, ab_0, tb_0 = self.cubic_motion_planning(self.initial_positions, self.grasp_position, time_interval[0], time_interval[1], self.tm*int(time_interval[1]-time_interval[0]))
        qg_0, vg_0, ag_0, tg_0 = self.cubic_motion_planning(self.initial_joint, self.initial_joint, time_interval[0], time_interval[1], self.tm*int(time_interval[1]-time_interval[0]))
        qb_1, vb_1, ab_1, tb_1 = self.cubic_motion_planning(self.grasp_position, self.grasp_position, time_interval[1], time_interval[2], self.tm*int(time_interval[2]-time_interval[1]))
        qg_1, vg_1, ag_1, tg_1 = self.cubic_motion_planning(self.initial_joint, self.grasp_joint, time_interval[1], time_interval[2], self.tm*int(time_interval[2]-time_interval[1]))
        qb_2, vb_2, ab_2, tb_2 = self.cubic_motion_planning(self.grasp_position, self.liftup_position, time_interval[2], time_interval[3], self.tm*int(time_interval[3]-time_interval[2]))
        qg_2, vg_2, ag_2, tg_2 = self.cubic_motion_planning(self.grasp_joint, self.grasp_joint, time_interval[2], time_interval[3], self.tm*int(time_interval[3]-time_interval[2]))
        
        return np.concatenate((qb_0, qb_1, qb_2)), np.concatenate((vb_0, vb_1, vb_2)), np.concatenate((ab_0, ab_1, ab_2)), np.concatenate((tb_0, tb_1, tb_2)), np.concatenate((qg_0, qg_1, qg_2)), np.concatenate((vg_0, vg_1, vg_2)), np.concatenate((ag_0, ag_1, ag_2)), np.concatenate((tg_0, tg_1, tg_2))
        