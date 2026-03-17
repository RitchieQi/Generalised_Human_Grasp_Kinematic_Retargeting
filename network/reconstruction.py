import numpy as np
import trimesh
import skimage.measure
import time
import torch
import os
import sys
import yaml
import json
from loguru import logger
from network.networks import kinematic_embedding
from network.solver import icp_ts
_manopth_root = os.path.join(os.path.dirname(__file__), 'manopth')
if os.path.isdir(_manopth_root) and _manopth_root not in sys.path:
    sys.path.append(_manopth_root)
try:
    from manopth.manolayer import ManoLayer
except ModuleNotFoundError:
    from manopth.manopth.manolayer import ManoLayer
import traceback
import os.path as osp

_HAND_FACES_CACHE = None
NETWORK_RECON_SCALE = 6.2
MESH_RECON_SCALE = 2.0/NETWORK_RECON_SCALE


def _results_root():
    return osp.join(osp.dirname(__file__), "results")


def _resolve_data_root():
    candidates = [
        osp.join(osp.dirname(__file__), '..', 'dataset', 'data'),
        osp.join(osp.dirname(__file__), '..', 'CtcSDF', 'data'),
    ]
    for p in candidates:
        if osp.isdir(p):
            return p
    return candidates[0]


def _get_hand_faces():
    global _HAND_FACES_CACHE
    if _HAND_FACES_CACHE is not None:
        return _HAND_FACES_CACHE

    candidates = [
        osp.join(osp.dirname(__file__), '..', 'dataset', 'data', 'closed_fmano.npy'),
        osp.join(osp.dirname(__file__), '..', 'CtcSDF', 'closed_fmano.npy'),
    ]
    for face_path in candidates:
        if osp.exists(face_path):
            _HAND_FACES_CACHE = np.load(face_path)
            return _HAND_FACES_CACHE

    mano_root = osp.join(osp.dirname(__file__), 'manopth', 'mano', 'models')
    if not osp.isdir(mano_root):
        mano_root = osp.join(osp.dirname(__file__), 'manopth', 'manopth', 'mano', 'models')
    mano_layer = ManoLayer(
        ncomps=45,
        side='right',
        mano_root=mano_root,
        use_pca=False,
        flat_hand_mean=True,
    )
    _HAND_FACES_CACHE = mano_layer.th_faces.detach().cpu().numpy()
    return _HAND_FACES_CACHE


def reconstruct(
    filename,
    obj_sdf_decoder,
    latent_vec,
    metas,
    hand_pose_results,
    obj_pose_results,
    point_batch_size=2**18,
    obj_point_latent=6,
    recon_scale=6.2,
    data_dir=None,
    results_root=None,
    recon_mode="icp",
):
    ply_filename_obj = filename[0] + "_obj"
    ply_filename_hand = filename[0] + "_hand"
    obj_sdf_decoder.eval()
    data_root = _resolve_data_root()
    data_dir = osp.join(data_root, 'mesh_data', 'mesh_obj')
    data_dir_hand = osp.join(data_root, 'mesh_data', 'mesh_hand')

    N = 128 #mesh resolution
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N-1)
    
    overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
    samples = torch.zeros(N**3, 5)
    
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3
    samples.requires_grad = False
    query_device = latent_vec.device if isinstance(latent_vec, torch.Tensor) else torch.device("cpu")
    obj_pose_results = {
        k: (v.to(query_device) if isinstance(v, torch.Tensor) else v)
        for k, v in obj_pose_results.items()
    }
    head = 0
    max_batch = point_batch_size
    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].to(query_device)
        obj_sample_subset = kinematic_embedding(obj_point_latent, recon_scale, sample_subset, sample_subset.shape[0], obj_pose_results, 'obj')
        obj_sample_subset = obj_sample_subset.reshape((-1, obj_point_latent))
    
        sdf_obj = decode_sdf(obj_sdf_decoder, latent_vec, obj_sample_subset, 'obj')
        samples[head : min(head + max_batch, num_samples), 4] = sdf_obj.squeeze(1).detach().cpu()

        head += max_batch
    
    sdf_values_obj = samples[:, 4]
    sdf_values_obj = sdf_values_obj.reshape((N, N, N))
    
    new_voxel_size, new_origin = get_higher_res_cube(sdf_values_obj, voxel_origin, voxel_size, N)
    samples_hr = torch.zeros(N**3, 5)
    samples_hr[:, 2] = overall_index % N
    samples_hr[:, 1] = (overall_index.long() / N) % N
    samples_hr[:, 0] = ((overall_index.long() / N) / N) % N
    samples_hr[:, 0] = (samples_hr[:, 0] * new_voxel_size) + new_origin[0]
    samples_hr[:, 1] = (samples_hr[:, 1] * new_voxel_size) + new_origin[1]
    samples_hr[:, 2] = (samples_hr[:, 2] * new_voxel_size) + new_origin[2]

    samples_hr.requires_grad = False
    
    head = 0
    while head < num_samples:
        sample_subset = samples_hr[head : min(head + max_batch, num_samples), 0:3].to(query_device)
        obj_sample_subset = kinematic_embedding(obj_point_latent, recon_scale, sample_subset, sample_subset.shape[0], obj_pose_results, 'obj')
        obj_sample_subset = obj_sample_subset.reshape((-1, obj_point_latent))
        sdf_obj = decode_sdf(obj_sdf_decoder, latent_vec, obj_sample_subset, 'obj')
        samples_hr[head : min(head + max_batch, num_samples), 4] = sdf_obj.squeeze(1).detach().cpu()
        head += max_batch
    
    sdf_values_obj = samples_hr[:, 4]
    sdf_values_obj = sdf_values_obj.reshape((N, N, N))
    voxel_size = new_voxel_size
    voxel_origin = new_origin.tolist()
    
    if recon_mode not in {"icp", "free"}:
        raise ValueError(f"Unknown recon_mode '{recon_mode}'. Use one of: icp, free.")

    use_icp = recon_mode == "icp"
    obj_dir = "mesh" if use_icp else "mesh_free"
    hand_dir = "mesh_hand" if use_icp else "mesh_hand_free"
    convert_verts_to_ply(
        sdf_result_dir=osp.join(results_root, hand_dir),
        ply_filename_out=ply_filename_hand + ".ply",
        testset_hand_source=data_dir_hand,
        recon_scale=recon_scale,
        use_icp=use_icp,
        offset=None,
        scale=None,
    )
    convert_sdf_samples_to_ply(
        sdf_tensor=sdf_values_obj.data.cpu(),
        voxel_origin=voxel_origin,
        voxel_size=voxel_size,
        optim=False,
        sdf_result_dir=osp.join(results_root, obj_dir),
        ply_filename_out=ply_filename_obj + ".ply",
        testset_obj_source=data_dir,
        recon_scale=recon_scale,
        metas=metas,
        use_icp=use_icp,
        fixed_scale=MESH_RECON_SCALE if not use_icp else None,
        offset=None,
        scale=None,
    )

    

    


def decode_sdf(sdf_decoder, latent_vector, points, mode):
    # points: N x points_dim_embeddding
    num_points = points.shape[0]
    latent_repeat = latent_vector.expand(num_points, -1)
    inputs = torch.cat([latent_repeat, points], 1)

    if mode == 'hand':
        sdf_val, predicted_class = sdf_decoder(inputs)
        return sdf_val, predicted_class
    else:
        sdf_val, _ = sdf_decoder(inputs)
        return sdf_val
    
def get_higher_res_cube(sdf_values_obj, voxel_origin, voxel_size, N):

    indices = torch.nonzero(sdf_values_obj < 0).float()
    if indices.shape[0] == 0:
        min_obj = torch.Tensor([0., 0., 0.])
        max_obj = torch.Tensor([0., 0., 0.])
    else:
        x_min_obj = torch.min(indices[:,0])
        y_min_obj = torch.min(indices[:,1])
        z_min_obj = torch.min(indices[:,2])
        min_obj = torch.Tensor([x_min_obj, y_min_obj, z_min_obj])

        x_max_obj = torch.max(indices[:,0])
        y_max_obj = torch.max(indices[:,1])
        z_max_obj = torch.max(indices[:,2])
        max_obj = torch.Tensor([x_max_obj, y_max_obj, z_max_obj])


    min_index = min_obj
    max_index = max_obj

    # Buffer 2 voxels each side
    new_cube_size = (torch.max(max_index - min_index) + 4) * voxel_size

    new_voxel_size = new_cube_size / (N - 1)
    # [z,y,x]
    new_origin = (min_index - 2 ) * voxel_size - 1.0  # (-1,-1,-1) origin

    return new_voxel_size, new_origin

def convert_sdf_samples_to_ply(
    sdf_tensor,
    voxel_origin,
    voxel_size,
    optim,
    sdf_result_dir,
    ply_filename_out,
    recon_scale,
    testset_obj_source,
    metas,
    use_icp=True,
    fixed_scale=None,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    hand_pose_result_dir = osp.join(_results_root(), 'hand_pose_results')
    os.makedirs(sdf_result_dir, exist_ok=True)
    sdf_tensor = sdf_tensor.numpy()
    # obj_pose_result_dir = osp.join(osp.dirname(__file__), 'obj_pose_results')
    # with open(os.path.join(obj_pose_result_dir, '_'.join(ply_filename_out.split('_')[:-1]) + '.json'), 'r') as f:
    #     data_ = json.load(f)
    # cam_extr = np.array(data_['cam_extr'], dtype=np.float32)
    # obj_trans = np.array(data_['global_trans'], dtype=np.float32)[:3,3].reshape(1, -1)
    # obj_center = np.array(data_['center'], dtype=np.float32).reshape(1, -1)
    # print('obj_trans_cam', (cam_extr @ obj_trans.transpose(1, 0)).transpose(1, 0))
    # print('obj_center_cam', (cam_extr @ obj_center.transpose(1, 0)).transpose(1, 0))
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(sdf_tensor, level=0.0, spacing=[voxel_size] * 3)
        with open(os.path.join(hand_pose_result_dir, '_'.join(ply_filename_out.split('_')[:-1]) + '.json'), 'r') as f:
            data = json.load(f)
        cam_extr = np.array(data['cam_extr'], dtype=np.float32)
        verts = (cam_extr @ verts.transpose(1, 0)).transpose(1, 0)

    except:
        traceback.print_exc()
        logger.warning("Cannot reconstruct mesh from '{}'".format(ply_filename_out))
        fallback_scale = 1.0 if use_icp else MESH_RECON_SCALE
        return None, None, np.array([0,0,0]), np.array([fallback_scale], dtype=np.float32)
    mesh_points_local = np.zeros_like(verts)
    mesh_points_local[:, 0] = voxel_origin[0] + verts[:, 0]
    mesh_points_local[:, 1] = voxel_origin[1] + verts[:, 1]
    mesh_points_local[:, 2] = voxel_origin[2] + verts[:, 2]
    # apply additional offset and scale
    # if scale is not None:
    #     mesh_points = mesh_points * scale
    # if offset is not None:
    #     mesh_points = mesh_points + offset
    if offset is None:
        # Object SDF is decoded in a hand-centered frame.
        # Place reconstructed object back by wrist joint (joint[0]) in camera frame.
        joints_0 = np.array(data['joints'], dtype=np.float32)[0].reshape(1, -1)
        joints_0 = (cam_extr @ joints_0.transpose(1, 0)).transpose(1, 0)
        offset = joints_0
    if use_icp:
        mesh_points = mesh_points_local + offset
    else:
        mesh_scale = MESH_RECON_SCALE if fixed_scale is None else float(fixed_scale)
        mesh_points = mesh_points_local * mesh_scale + offset

    pred_mesh = trimesh.Trimesh(vertices=mesh_points, faces=faces, process=False)
    split_mesh = trimesh.graph.split(pred_mesh)
    if len(split_mesh) > 1:
        max_area = -1
        final_mesh = split_mesh[0]
        for per_mesh in split_mesh:
            if per_mesh.area > max_area:
                max_area = per_mesh.area
                final_mesh = per_mesh
        pred_mesh = final_mesh
    if use_icp:
        trans = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        out_scale = np.array([1.0], dtype=np.float32)
        try:
            gt_mesh = trimesh.load(
                os.path.join(testset_obj_source, '_'.join(ply_filename_out.split('_')[:-1]) + '.obj'),
                process=False,
            )
            icp_solver = icp_ts(pred_mesh, gt_mesh)
            icp_solver.sample_mesh(30000, 'both')
            icp_solver.run_icp_f(max_iter=100)
            icp_solver.export_source_mesh(os.path.join(sdf_result_dir, ply_filename_out))
            trans, out_scale = icp_solver.get_trans_scale()
            return verts, faces, np.asarray(trans, dtype=np.float32), np.asarray(out_scale, dtype=np.float32).reshape(1)
        except Exception:
            traceback.print_exc()
            logger.warning(
                "Cannot load ground truth mesh from '{}'".format(
                    os.path.join(testset_obj_source, '_'.join(ply_filename_out.split('_')[:-1]) + '.obj')
                )
            )
            pred_mesh.export(os.path.join(sdf_result_dir, ply_filename_out))
            return verts, faces, trans, out_scale

    out_scale = MESH_RECON_SCALE if fixed_scale is None else float(fixed_scale)
    pred_mesh.export(os.path.join(sdf_result_dir, ply_filename_out))
    return verts, faces, np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([out_scale], dtype=np.float32)

def convert_verts_to_ply(sdf_result_dir, ply_filename_out, testset_hand_source, recon_scale, use_icp=True, offset=None, scale=None):
    hand_pose_result_dir = osp.join(_results_root(), 'hand_pose_results')
    os.makedirs(sdf_result_dir, exist_ok=True)
    with open(os.path.join(hand_pose_result_dir, '_'.join(ply_filename_out.split('_')[:-1]) + '.json'), 'r') as f:
        data = json.load(f)
    cam_extr = np.array(data['cam_extr'], dtype=np.float32)
    verts = np.array(data['verts'], dtype=np.float32)
    rot_center = np.array(data['rot_center'], dtype=np.float32).reshape(1, -1)
    verts_cam = (cam_extr @ verts.transpose(1, 0)).transpose(1, 0)

    joints_0 = np.array(data['joints'], dtype=np.float32)[0].reshape(1, -1)

    joints_0 = (cam_extr @ joints_0.transpose(1, 0)).transpose(1, 0)

    # verts_cam = verts
    faces = _get_hand_faces()
    pred_hand_mesh = trimesh.Trimesh(vertices=verts_cam, faces=faces, process=False)
    if use_icp:
        try:
            gt_mesh = trimesh.load(
                os.path.join(testset_hand_source, '_'.join(ply_filename_out.split('_')[:-1]) + '.obj'),
                process=False,
            )
            icp_solver = icp_ts(pred_hand_mesh, gt_mesh)
            icp_solver.sample_mesh(30000, 'both')
            icp_solver.run_icp_f(max_iter=100)
            icp_solver.export_source_mesh(os.path.join(sdf_result_dir, ply_filename_out))
            trans, out_scale = icp_solver.get_trans_scale()
            return verts_cam + trans, faces, np.asarray(trans, dtype=np.float32), np.asarray(out_scale, dtype=np.float32).reshape(1)
        except Exception:
            logger.warning(
                "Cannot load ground truth mesh from '{}'".format(
                    os.path.join(testset_hand_source, '_'.join(ply_filename_out.split('_')[:-1]) + '.obj')
                )
            )
            pred_hand_mesh.export(os.path.join(sdf_result_dir, ply_filename_out))
            return verts_cam, faces, np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)
    pred_hand_mesh.export(os.path.join(sdf_result_dir, ply_filename_out))
    return verts_cam, faces, np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)
