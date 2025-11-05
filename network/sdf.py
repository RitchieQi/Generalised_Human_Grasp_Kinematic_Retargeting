import os
import torch
from model import get_model
from networks import kinematic_embedding
from reconstruction import decode_sdf

class DeepSDF_object:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = get_model(is_train=False, batch=1, num_sample_points=2048, clamp_dist=0.05)
        self.model.to(self.device)
        ckpt = torch.load(model_path)
        self.model.load_state_dict(ckpt['network'])
        self.model.eval()
        self.obj_point_latent = 6
        self.recon_scale = 6.2
        
    
    def distance(self, input_img, metas, x):
        sdf_feat, hand_pose_results, obj_pose_results = self.model(input_img, None, metas, mode='test')
        obj_sample_subset = kinematic_embedding(self.obj_point_latent, self.recon_scale, x, x.shape[0], obj_pose_results, 'obj')
        obj_sample_subset = obj_sample_subset.to(self.device)
        obj_sample_subset = obj_sample_subset.reshape((-1, self.obj_point_latent))
        
        sdf_obj = decode_sdf(self.model.module.obj_sdf_head,sdf_feat, obj_sample_subset, 'obj')
        
        return sdf_obj
    
        
    def gradient(self, x, distance, retain_graph=False, create_graph=False, allow_unused=False):
        return torch.autograd.grad([distance.sum()], [x], retain_graph=retain_graph, create_graph=create_graph, allow_unused=allow_unused)[0]

        
        
        