import mujoco as m
import torch
import trimesh as tm
import os
import pytorch_kinematics as pk
import numpy as np
from scipy.spatial.transform import Rotation as R

def euler_to_mat(euler, degrees=True):
    euler = euler.cpu().numpy()
    r = R.from_euler('xyz', euler, degrees=degrees)
    return torch.tensor(r.as_matrix())

def mat_to_euler(mat, degrees=True):
    mat = mat[0].cpu().numpy()
    r = R.from_matrix(mat)
    return torch.tensor(r.as_euler('xyz', degrees=degrees).copy())

def mat_to_quat(mat):
    mat = mat.cpu().numpy()
    r = R.from_matrix(mat)
    reorder = torch.tensor([3, 0, 1, 2], dtype=torch.long)
    return torch.tensor(r.as_quat().copy())[:, reorder]

def mat2axisang(mat):
    quat = mat_to_quat(mat)
    quat_norm = torch.norm(quat, p=2, dim=1)
    quat= torch.div(quat, quat_norm)
    angle = 2*torch.acos(quat[:, 0])
    sin_theta = torch.sqrt(1 - quat[:, 0]**2).clamp(min=1e-6)
    axis = quat[:, 1:] / sin_theta.unsqueeze(1)
    axisang = axis * angle.unsqueeze(1)
    return axisang

def mat2axisang_(mat):
    quat = mat_to_quat(mat)
    quat_norm = torch.norm(quat, p=2, dim=1)
    quat= torch.div(quat, quat_norm)

def compute_geodesic_distance_from_two_matrices(m1, m2, device):
    batch=m1.shape[0]
    m = torch.matmul(m1, m2.transpose(1,2)) #batch*3*3
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).to(device=device)) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).to(device=device))*-1 )
    theta = torch.acos(cos)
    return theta

def axis2quat(axisang):
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5    
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    return quat

def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
    return out

def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v/v_mag
    return v

def robust_compute_rotation_matrix_from_ortho6d(poses):
    """
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    y = normalize_vector(y_raw)  # batch*3
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    # Their scalar product should be small !
    # assert torch.einsum("ij,ij->i", [x, y]).abs().max() < 0.00001
    z = normalize_vector(cross_product(x, y))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    # Check for reflection in matrix ! If found, flip last vector TODO
    # assert (torch.stack([torch.det(mat) for mat in matrix ])< 0).sum() == 0
    return matrix

#TODO: need a data loader to load the data
def load_npz(path):
    npz = np.load(path, allow_pickle=True)
    obj_center = npz['obj_center']
    joints = npz['joints']
    return obj_center, joints

def get_fingertip_pos(trans3ds):
    _pos = []
    for i, name in enumerate(trans3ds):
        pos, rot = quat_pos_from_transform3d(trans3ds[name])
        _pos.append(pos)
    _pos = torch.stack(_pos, dim=1)
    return _pos

def quat_pos_from_transform3d(trans3d):
    m = trans3d.get_matrix()
    pos = m[:, :3, 3]
    rot = pk.matrix_to_quaternion(m[:, :3, :3])
    return pos, rot

def set_global_seed(seed=42):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def rotation_vectors(a,b):
    """
    input:
    --------------------------------
    a: (3,)
    b: (3,)

    output:
    --------------------------------
    T: (3,3)

    Return the rotation matrix that rotates vector a to vector b

    """
    #orthogonal vector
    #Note this will equal to 0 if a and b are parallel or anti-parallel
    cross = torch.cross(a, b)


    sin = (cross.T/torch.linalg.norm(cross)).T
    cos = torch.dot(a, b)
    skew_symmetric_constructor = torch.tensor(np.array([[0,0,0,0,0,-1,0,1,0], [0,0,1,0,0,0,-1,0,0], [0,-1,0,1,0,0,0,0,0]])).float()
    I = torch.eye(3)
    cross_i = torch.matmul(cross, skew_symmetric_constructor).reshape([3,3])
    cross_i_2 = torch.matmul(cross_i, cross_i)
    T = I + cross_i + cross_i_2 * (1 / (1 + cos))
    return T

def multi_rotation_vectors_batch(a,b):
    """
    input:
    --------------------------------
    a: B*N*3
    b: B*N*3

    output:
    --------------------------------
    T: B*N*3*3

    Return the rotation matrix that rotates multiple vectors a to multiple vectors b

    """
    B, N, _ = a.shape
    cross = torch.cross(a, b, dim=2)
    sin = cross / torch.linalg.norm(cross, dim=2, keepdim=True)
    cos = torch.sum(a * b, dim=2, keepdim=True)
    skew_symmetric_constructor = torch.tensor(np.array([[0,0,0,0,0,-1,0,1,0], [0,0,1,0,0,0,-1,0,0], [0,-1,0,1,0,0,0,0,0]])).float().to(a.device)
    xi_cross = torch.matmul(cross, skew_symmetric_constructor).reshape([B,N,3,3])
    I = torch.eye(3).view(1, 1, 3, 3).repeat(B, N, 1, 1).to(a.device)
    xi_cross_2 = torch.matmul(xi_cross, xi_cross)
    T = I + xi_cross + xi_cross_2 * (1 / (1 + cos).unsqueeze(-1))
    return T



def linearized_cone(normal, w):
    """
    input:
    --------------------------------
    normal: B x N x 3
    w: B x N x 4

    output:
    --------------------------------
    f: B x N x 3

    Return the weights of the 4-edge friction pyramid
    """
    B = normal.shape[0]
    N = normal.shape[1]
    z = torch.tensor([0., 0., 1.]).view(1, 1, 3).repeat([B, N, 1])
    # construct 4-edge friction pyramid 
    mu = 0.1
    e1 = torch.tensor([mu, 0, 1]).view(1, 1, 3).repeat([B, N, 1])
    e2 = torch.tensor([0, mu, 1]).view(1, 1, 3).repeat([B, N, 1])
    e3 = torch.tensor([mu, 0, 1]).view(1, 1, 3).repeat([B, N, 1])
    e4 = torch.tensor([0, mu, 1]).view(1, 1, 3).repeat([B, N, 1])

    # f = w_0e_0 + , w > 0
    # fe^-1 > 0
    E = torch.stack([e1, e2, e3, e4], -2) # B x N x 4 x 3
    T = multi_rotation_vectors_batch(z, normal) # B x N x 3 x 3

    #apple T to E
    E_transed = torch.matmul(T, E.transpose(-2,-1)).transpose(-2,-1) # B x N x 4 x 3

    #E_pinv = torch.linalg.pinv(E_transed) # B x N x 4 x 3

    #print(E_pinv.shape)
    print(E_transed.shape)
    print(w.shape)
    f = torch.matmul(w.unsqueeze(-2), E_transed).squeeze()
    w_edges = (E_transed * w.unsqueeze(-1)).reshape(B, N*4, 3)

    
    

    return f, w_edges

