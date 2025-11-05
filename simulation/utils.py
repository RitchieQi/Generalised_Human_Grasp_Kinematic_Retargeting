
import os
osp = os.path
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc
import numpy as np
from typing import List, Optional, Tuple, Union, Iterator, Any, Dict
from scipy.spatial.transform import Rotation as R

import os
def rodrigues(axisang):
    """rewrite from the torch version batch_rodrigues from mano repo
       The quaternions are written in scaler-last manner to be compatible with bullet3
    """
    axisang_norm = np.linalg.norm(axisang + 1e-8, ord=2)
    axisang_normalized = axisang/axisang_norm
    angle = axisang_norm * 0.5
    
    v_cos = np.cos(angle)
    v_sin = np.sin(angle)
    quat = np.hstack([v_sin*axisang_normalized, v_cos]) #xyzw
    quat_r = R.from_quat(quat)
    mat = quat_r.as_matrix()
    return quat, mat

def quat_mult(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    x = x1 * w2 + w1 * x2 + y1 * z2 - z1 * y2
    y = y1 * w2 + w1 * y2 + z1 * x2 - x1 * z2
    z = z1 * w2 + w1 * z2 + x1 * y2 - y1 * x2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w])

def quaternion_conjugate(q):
    """
    Computes the conjugate of a quaternion (scalar-last format).
    Args:
        q: Quaternion [x, y, z, w]
    Returns:
        Conjugate quaternion [-x, -y, -z, w]
    """
    x, y, z, w = q
    return np.array([-x, -y, -z, w])

def rotate_vector(vector, quaternion):
    """
    Rotates a 3D vector using a quaternion (scalar-last format).
    Args:
        vector: 3D vector [vx, vy, vz]
        quaternion: Rotation quaternion [x, y, z, w]
    Returns:
        Rotated 3D vector [vx', vy', vz']
    """
    q = normalize_quaternion(quaternion)
    # Convert vector to quaternion (q_v)
    q_v = np.array([*vector, 0])  # Scalar part is 0

    # Compute rotated quaternion: q_rotated = q * q_v * q_conjugate
    q_conjugate = quaternion_conjugate(quaternion)
    q_rotated = quat_mult(quat_mult(quaternion, q_v), q_conjugate)

    # Extract the rotated vector (ignore the scalar part)
    return q_rotated[:3]

def cubic_motion_planning(initial_positions: np.ndarray, target_positions: np.ndarray, t0: int, tf: int, num_points: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

def rotate_vector_inverse(vector, quaternion):
    """
    Rotate a vector using the inverse of a quaternion
    """
    #Normalize the quaternion
    q = normalize_quaternion(quaternion)
    
    q_v = np.array([*vector, 0])
    
    q_conjugate = quaternion_conjugate(q)
    
    q_rotated = quat_mult(quat_mult(q_conjugate, q_v), q)
    
    return q_rotated[:3]

def normalize_quaternion(q):
    """Normalize a quaternion."""
    norm = np.linalg.norm(q)
    return q / norm

def euler_to_mat(euler, degrees=True):
    #euler = euler.cpu().numpy()
    r = R.from_euler('xyz', euler, degrees=degrees).as_matrix()
    return np.array(r)

def reverse_rotate_mat(v, q):
    q = normalize_quaternion(q)
    r_m = R.from_quat(q).as_matrix()
    return np.dot(r_m.T, v)
    