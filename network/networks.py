#@File        :resnet.py
#@Date        :2022/04/07 10:19:40
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import torch
import torch.nn as nn
import os
import sys
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
try:
    from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
except ImportError:
    ResNet18_Weights = ResNet34_Weights = ResNet50_Weights = ResNet101_Weights = ResNet152_Weights = None
_manopth_root = os.path.join(os.path.dirname(__file__), 'manopth')
if os.path.isdir(_manopth_root) and _manopth_root not in sys.path:
    sys.path.append(_manopth_root)
try:
    from manopth.manolayer import ManoLayer
except ModuleNotFoundError:
    from manopth.manopth.manolayer import ManoLayer
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import numpy as np
class ResNetBackbone(nn.Module):
    def __init__(self, resnet_type):
        resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
		       34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
		       50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
		       101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
		       152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}
        block, layers, channels, name = resnet_spec[resnet_type]
        
        self.name = name
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, return_inter=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if return_inter:
            inter_feat = x

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if return_inter:
            return x, inter_feat
        else:
            return x

    def init_weights(self):
        weights_map = {
            'resnet18': (resnet18, ResNet18_Weights.IMAGENET1K_V1 if ResNet18_Weights is not None else None),
            'resnet34': (resnet34, ResNet34_Weights.IMAGENET1K_V1 if ResNet34_Weights is not None else None),
            'resnet50': (resnet50, ResNet50_Weights.IMAGENET1K_V1 if ResNet50_Weights is not None else None),
            'resnet101': (resnet101, ResNet101_Weights.IMAGENET1K_V1 if ResNet101_Weights is not None else None),
            'resnet152': (resnet152, ResNet152_Weights.IMAGENET1K_V1 if ResNet152_Weights is not None else None),
        }
        model_fn, weights = weights_map[self.name]
        if weights is not None:
            org_resnet = model_fn(weights=weights).state_dict()
        else:
            org_resnet = model_fn(pretrained=True).state_dict()
        # drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
        org_resnet.pop('fc.weight', None)
        org_resnet.pop('fc.bias', None)
        self.load_state_dict(org_resnet)

class UNet(nn.Module):
    def __init__(self, inplanes, outplanes, num_stages, use_final_layers=False):
        super(UNet, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.num_stages = num_stages
        self.use_final_layers = use_final_layers
        self.deconv_layers = self._make_deconv_layer(self.num_stages)
        if self.use_final_layers:
            final_layers = []
            for i in range(3):
                final_layers.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
                final_layers.append(nn.BatchNorm2d(self.outplanes))
                final_layers.append(nn.ReLU(inplace=True))
            self.final_layers = nn.Sequential(*final_layers)

    def _make_deconv_layer(self, num_stages):
        layers = []
        for i in range(num_stages):
            layers.append(nn.ConvTranspose2d(in_channels=self.inplanes, out_channels=self.outplanes, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False))
            layers.append(nn.BatchNorm2d(self.outplanes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = self.outplanes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.deconv_layers(x)
        if self.use_final_layers:
            x = self.final_layers(x)
        return x

class SDFHead(nn.Module):
    def __init__(self, sdf_latent, point_latent, dims, dropout, dropout_prob, norm_layers, latent_in, use_cls_hand=False, num_class=0):
        super(SDFHead, self).__init__()
        self.sdf_latent = sdf_latent
        self.point_latent = point_latent
        self.dims = [self.sdf_latent + self.point_latent] + dims + [1]
        self.num_layers = len(self.dims)
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.use_cls_hand = use_cls_hand
        self.num_class = num_class

        self.relu = nn.ReLU()
        self.th = nn.Tanh()

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in self.latent_in:
                out_dim = self.dims[layer + 1] - self.dims[0]
            else:
                out_dim = self.dims[layer + 1]

            if layer in self.norm_layers:
                setattr(self, "lin" + str(layer), nn.utils.weight_norm(nn.Linear(self.dims[layer], out_dim)),)
            else:
                setattr(self, "lin" + str(layer), nn.Linear(self.dims[layer], out_dim))

            # classifier
            if self.use_cls_hand and layer == self.num_layers - 2:
                self.classifier_head = nn.Linear(self.dims[layer], self.num_class)

    def forward(self, input, pose_results=None):
        latent = input
       
        predicted_class = None
        for layer in range(0, self.num_layers - 1):
            if self.use_cls_hand and layer == self.num_layers - 2:
                predicted_class = self.classifier_head(latent)

            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                latent = torch.cat([latent, input], 1)
            latent = lin(latent)

            if layer < self.num_layers - 2:
                latent = self.relu(latent)
                if layer in self.dropout:
                    latent = F.dropout(latent, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            latent = self.th(latent)

        return latent, predicted_class

class ManoHead(nn.Module):
    def __init__(self, ncomps=15, base_neurons=[512, 512, 512], center_idx=0, use_shape=True, use_pca=True, mano_root=os.path.join(os.path.dirname(__file__), 'manopth','mano','models'), depth=False):
        super(ManoHead, self).__init__()
        self.ncomps = ncomps
        self.base_neurons = base_neurons
        self.center_idx = center_idx
        self.use_shape = use_shape
        self.use_pca = use_pca
        self.depth = depth

        if self.use_pca:
            # pca comps + 3 global axis-angle params
            mano_pose_size = ncomps + 3
        else:
            # 15 joints + 1 global rotations, 9 comps per rot
            mano_pose_size = 16 * 9

        # Base layers
        base_layers = []
        for layer_idx, (inp_neurons, out_neurons) in enumerate(zip(self.base_neurons[:-1], self.base_neurons[1:])):
            base_layers.append(nn.Linear(inp_neurons, out_neurons))
            base_layers.append(nn.ReLU())
        self.base_layers = nn.Sequential(*base_layers)

        # Pose layers
        self.pose_reg = nn.Linear(self.base_neurons[-1], mano_pose_size)

        # Shape layers
        if self.use_shape:
            self.shape_reg = torch.nn.Sequential(nn.Linear(self.base_neurons[-1], 10))
        
        if self.depth:
            depth_layers = []
            trans_neurons = [512, 256]
            for layer_idx, (inp_neurons, out_neurons) in enumerate(zip(trans_neurons[:-1], trans_neurons[1:])):
                depth_layers.append(nn.Linear(inp_neurons, out_neurons))
                depth_layers.append(nn.ReLU())
            depth_layers.append(nn.Linear(trans_neurons[-1], 3))
            self.depth_layers = nn.Sequential(*depth_layers)
        
        # Mano layers
        self.mano_layer_right = ManoLayer(
            ncomps=self.ncomps,
            center_idx=self.center_idx,
            side="right",
            mano_root=mano_root,
            use_pca=self.use_pca,
            flat_hand_mean=False
        )

    def forward(self, inp):
        mano_features = self.base_layers(inp)
        pose = self.pose_reg(mano_features)

        scale_trans = None
        if self.depth:
            scale_trans = self.depth_layers(inp)

        if self.use_pca:
            mano_pose = pose
        else:
            mano_pose = pose.reshape(pose.shape[0], 16, 3, 3)

        if self.use_shape:
            shape = self.shape_reg(mano_features)
        else:
            shape = None

        if mano_pose is not None and shape is not None:
            verts, joints, poses, global_trans, rot_center, _ = self.mano_layer_right(mano_pose, th_betas=shape, root_palm=False)
        
        valid_idx = torch.ones((inp.shape[0], 1), device=inp.device).long()
        mean_pose = torch.from_numpy(np.array(self.mano_layer_right.smpl_data['hands_mean'], dtype=np.float32)).to(inp.device)

        results = {"verts": verts, "joints": joints, "shape": shape, "pcas": mano_pose, "pose": poses,  "global_trans":global_trans, "rot_center": rot_center, "scale_trans": scale_trans, "vis": valid_idx, "mean_pose": mean_pose}

        return results
    
    
def kinematic_embedding(obj_point_latent, recon_scale, input_points, num_points_per_scene, pose_results, mode):

    assert obj_point_latent in [6, 9, 69, 72], 'please set a right object embedding size'

    input_points = input_points.reshape((-1, num_points_per_scene, 3))
    batch_size = input_points.shape[0]
    try:
        inv_func = torch.linalg.inv
    except:
        inv_func = torch.inverse
    

    xyz = (input_points * 2 / recon_scale)
    obj_trans = pose_results['global_trans']
    homo_xyz_obj = homoify(xyz) #cude
    inv_obj_trans = inv_func(obj_trans)
    inv_homo_xyz_obj = torch.matmul(inv_obj_trans, homo_xyz_obj.transpose(2, 1)).transpose(2, 1)
    inv_xyz_obj = dehomoify(inv_homo_xyz_obj)
    try:
        hand_trans = pose_results['wrist_trans']
        xyz_mano = xyz
        homo_xyz_mano = homoify(xyz_mano)
        inv_hand_trans = inv_func(hand_trans)
    except:
        pass

    if obj_point_latent == 6:
        point_embedding = torch.cat([xyz, inv_xyz_obj], 2)
    
    if obj_point_latent == 9:
        inv_homo_xyz_mano = torch.matmul(inv_hand_trans, homo_xyz_mano.transpose(1, 2)).transpose(1, 2)
        inv_xyz_mano = dehomoify(inv_homo_xyz_mano)
        point_embedding = torch.cat([xyz, inv_xyz_obj, inv_xyz_mano], 2)

    if obj_point_latent == 69:
        inv_xyz_joint = [xyz, inv_xyz_obj]
        for i in range(21):
            inv_xyz_joint.append(xyz - pose_results['joints'][:, [i], :])
        point_embedding = torch.cat(inv_xyz_joint, 2)
    
    if obj_point_latent == 72:
        inv_homo_xyz_mano = torch.matmul(inv_hand_trans, homo_xyz_mano.transpose(1, 2)).transpose(1, 2)
        inv_xyz_mano = dehomoify(inv_homo_xyz_mano)
        inv_xyz_joint = [xyz, inv_xyz_obj, inv_xyz_mano]
        for i in range(21):
            inv_xyz_joint.append(xyz - pose_results['joints'][:, [i], :])
        point_embedding = torch.cat(inv_xyz_joint, 2)
                
        point_embedding = point_embedding.reshape((batch_size, num_points_per_scene, -1))
        point_embedding = point_embedding * recon_scale / 2

    return point_embedding

def homoify(points):
    """
    Convert a batch of points to homogeneous coordinates.
    Args:
        points: e.g. (B, N, 3) or (N, 3)
    Returns:
        homoified points: e.g., (B, N, 4)
    """
    points_dim = points.shape[:-1] + (1,)
    ones = points.new_ones(points_dim)

    return torch.cat([points, ones], dim=-1)


def dehomoify(points):
    """
    Convert a batch of homogeneous points to cartesian coordinates.
    Args:
        homogeneous points: (B, N, 4/3) or (N, 4/3)
    Returns:
        cartesian points: (B, N, 3/2)
    """
    return points[..., :-1] / points[..., -1:]
