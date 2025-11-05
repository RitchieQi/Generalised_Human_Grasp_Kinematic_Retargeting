import numpy as np
import trimesh
import skimage.measure
import time
import torch
import os
import yaml
import json
from loguru import logger
from networks import kinematic_embedding
from solver import icp_ts
from manopth.manolayer import ManoLayer
import traceback
import os.path as osp

data_dir = osp.join(osp.dirname(__file__), '..', 'CtcSDF', 'data','mesh_data', 'mesh_obj')
def reconstruct(filename, obj_sdf_decoder, latent_vec, metas, hand_pose_results, obj_pose_results, point_batch_size=2**18, obj_point_latent = 6, recon_scale = 6.2, data_dir = osp.join(osp.dirname(__file__), '..', 'CtcSDF', 'data','mesh_data', 'mesh_obj')):
    ply_filename_obj = filename[0] + "_obj"
    ply_filename_hand = filename[0] + "_hand"
    obj_sdf_decoder.eval()
    data_dir = osp.join(osp.dirname(__file__), '..', 'CtcSDF', 'data','mesh_data', 'mesh_obj')
    data_dir_hand = osp.join(osp.dirname(__file__), '..', 'CtcSDF', 'data','mesh_data', 'mesh_hand')

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
    head = 0
    max_batch = point_batch_size
    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()
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
        sample_subset = samples_hr[head : min(head + max_batch, num_samples), 0:3].cuda()
        obj_sample_subset = kinematic_embedding(obj_point_latent, recon_scale, sample_subset, sample_subset.shape[0], obj_pose_results, 'obj')
        obj_sample_subset = obj_sample_subset.reshape((-1, obj_point_latent))
        sdf_obj = decode_sdf(obj_sdf_decoder, latent_vec, obj_sample_subset, 'obj')
        samples_hr[head : min(head + max_batch, num_samples), 4] = sdf_obj.squeeze(1).detach().cpu()
        head += max_batch
    
    sdf_values_obj = samples_hr[:, 4]
    sdf_values_obj = sdf_values_obj.reshape((N, N, N))
    voxel_size = new_voxel_size
    voxel_origin = new_origin.tolist()
    
    verts,faces,trans, scale = convert_verts_to_ply(not_optim=False,
                                                   sdf_result_dir=osp.join(osp.dirname(__file__),'hmano_osdf','mesh_hand'),
                                                   ply_filename_out=ply_filename_hand + '.ply',
                                                   testset_hand_source=data_dir_hand,
                                                   recon_scale=recon_scale,
                                                   offset=None,
                                                   scale=None
                                                   )
    # print('object trans gt', metas['obj_transform'].squeeze().cpu().numpy()[:3,3])
    # print('offset', trans)
    convert_sdf_samples_to_ply(sdf_tensor=sdf_values_obj.data.cpu(), 
                               voxel_origin=voxel_origin, 
                               voxel_size=voxel_size, 
                               not_optim=True, 
                               sdf_result_dir=osp.join(osp.dirname(__file__),'hmano_osdf','mesh'), 
                               ply_filename_out=ply_filename_obj + '.ply', 
                               testset_obj_source=data_dir, 
                               recon_scale=recon_scale, 
                               metas=metas,
                               offset=None, scale=None)

    

    


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

def convert_sdf_samples_to_ply(sdf_tensor, voxel_origin, voxel_size, not_optim, sdf_result_dir, ply_filename_out, recon_scale, testset_obj_source, metas, offset=None, scale=None):
    """
    Convert sdf samples to .ply
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    hand_pose_result_dir = osp.join(osp.dirname(__file__), 'hmano_osdf', 'hand_pose_results')
    sdf_tensor = sdf_tensor.numpy()
    # obj_pose_result_dir = osp.join(osp.dirname(__file__), 'hmano_osdf', 'obj_pose_results')
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
        return None, None, np.array([0,0,0]), np.array([1 / recon_scale])
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_origin[2] + verts[:, 2]
    # apply additional offset and scale
    # if scale is not None:
    #     mesh_points = mesh_points * scale
    # if offset is not None:
    #     mesh_points = mesh_points + offset
    if offset is None:
        offset = np.array(metas['hand_center_3d'].cpu(), dtype=np.float32)
        offset = cam_extr @ offset.transpose(1, 0)
        offset = offset.transpose(1, 0)
        joints_0 = np.array(data['joints'], dtype=np.float32)[0].reshape(1, -1)
        joints_0 = (cam_extr @ joints_0.transpose(1, 0)).transpose(1, 0)
        residual = joints_0 - offset
        offset = offset + residual

        #offset = np.array([0, 0, 0])
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
    trans = np.array([0, 0, 0])
    scale = np.array([1])
    if not_optim:
        gt_mesh = trimesh.load(os.path.join(testset_obj_source, '_'.join(ply_filename_out.split('_')[:-1]) + '.obj'), process=False)
        try:
            icp_solver = icp_ts(pred_mesh, gt_mesh)
            icp_solver.sample_mesh(30000, 'both')
            icp_solver.run_icp_f(max_iter = 100)
            trans, scale = icp_solver.get_trans_scale()
            # print('obj_trans', trans)
            # print('obj_centre_before', pred_mesh.center_mass)
            pred_mesh.vertices = pred_mesh.vertices * scale + offset
            # print('obj_centre', pred_mesh.center_mass)
            pred_mesh.export(os.path.join(sdf_result_dir, ply_filename_out))
        except:
            traceback.print_exc()
            logger.warning("Cannot load ground truth mesh from '{}'".format(os.path.join(testset_obj_source, '_'.join(ply_filename_out.split('_')[:-1]) + '.obj')))
            pred_mesh.export(os.path.join(sdf_result_dir, ply_filename_out))
            return  None, None, np.array([0,0,0]), np.array([1 / recon_scale])
    else:
        gt_mesh = trimesh.load(os.path.join(testset_obj_source, '_'.join(ply_filename_out.split('_')[:-1]) + '.obj'), process=False)
        try:
            icp_solver = icp_ts(pred_mesh, gt_mesh)
            icp_solver.sample_mesh(30000, 'both')
            icp_solver.run_icp_f(max_iter = 100)
            icp_solver.export_source_mesh(os.path.join(sdf_result_dir, ply_filename_out))
            trans, scale = icp_solver.get_trans_scale()
            # print('obj_trans', trans)
            return verts, faces, trans, scale
        except:
            traceback.print_exc()
            logger.warning("Cannot load ground truth mesh from '{}'".format(os.path.join(testset_obj_source, '_'.join(ply_filename_out.split('_')[:-1]) + '.obj')))
            return None, None, np.array([0,0,0]), np.array([1 / recon_scale])

def convert_verts_to_ply(not_optim, sdf_result_dir, ply_filename_out, testset_hand_source, recon_scale, offset=None, scale=None):
    hand_pose_result_dir = osp.join(osp.dirname(__file__), 'hmano_osdf', 'hand_pose_results')
    with open(os.path.join(hand_pose_result_dir, '_'.join(ply_filename_out.split('_')[:-1]) + '.json'), 'r') as f:
        data = json.load(f)
    cam_extr = np.array(data['cam_extr'], dtype=np.float32)
    verts = np.array(data['verts'], dtype=np.float32)
    rot_center = np.array(data['rot_center'], dtype=np.float32).reshape(1, -1)
    verts_cam = (cam_extr @ verts.transpose(1, 0)).transpose(1, 0)

    joints_0 = np.array(data['joints'], dtype=np.float32)[0].reshape(1, -1)

    # joints_0 = (cam_extr @ (joints_0-rot_center).transpose(1, 0)).transpose(1, 0) + rot_center
    joints_0 = (cam_extr @ joints_0.transpose(1, 0)).transpose(1, 0)
    # verts_cam = (cam_extr @ (verts-joints_0).transpose(1, 0)).transpose(1, 0) + joints_0

    # verts_cam = verts
    faces = torch.LongTensor(np.load(osp.join(osp.dirname(__file__), '..', 'CtcSDF','closed_fmano.npy')))
    pred_hand_mesh = trimesh.Trimesh(vertices=verts_cam, faces=faces, process=False)
    gt_mesh = trimesh.load(os.path.join(testset_hand_source, '_'.join(ply_filename_out.split('_')[:-1]) + '.obj'), process=False)
    try:
        icp_solver = icp_ts(pred_hand_mesh, gt_mesh)
        icp_solver.sample_mesh(30000, 'both')
        icp_solver.run_icp_f(max_iter = 100)
        icp_solver.export_source_mesh(os.path.join(sdf_result_dir, ply_filename_out))
        trans, scale = icp_solver.get_trans_scale()
        # print('hand trans',trans)
        # print('joint_0', joints_0)
        # print('joints_0 + trans', trans+joints_0)
        return verts_cam, faces, trans+joints_0, np.array([1 / 3.1])
    except:
        logger.warning("Cannot load ground truth mesh from '{}'".format(os.path.join(testset_hand_source, '_'.join(ply_filename_out.split('_')[:-1]) + '.obj')))
        return None, None, joints_0, np.array([1 / 3.1])