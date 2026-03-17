import torch
try:
    from .model import get_model
    from .networks import kinematic_embedding
    from .reconstruction import decode_sdf
except ImportError:
    import os.path as osp
    import sys
    _THIS_DIR = osp.dirname(osp.abspath(__file__))
    if _THIS_DIR not in sys.path:
        sys.path.insert(0, _THIS_DIR)
    from model import get_model
    from networks import kinematic_embedding
    from reconstruction import decode_sdf

class DeepSDF_object:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = get_model(is_train=False, batch=1, num_sample_points=2048, clamp_dist=0.05)
        self.model.to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        state_dict = ckpt["network"]
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            stripped = {}
            for key, value in state_dict.items():
                if key.startswith("module."):
                    stripped[key[len("module."):]] = value
                else:
                    stripped[key] = value
            self.model.load_state_dict(stripped)
        self.model.eval()
        self.obj_point_latent = 6
        self.recon_scale = 6.2
        self._cached_input_img = None
        self._cached_metas = None
        self._cached_sdf_feat = None
        self._cached_obj_pose_results = None

    def _obj_sdf_head(self):
        if hasattr(self.model, "module"):
            return self.model.module.obj_sdf_head
        return self.model.obj_sdf_head

    def _as_input_dict(self, input_img):
        if isinstance(input_img, dict):
            return input_img
        return {"img": input_img}
        
    
    def distance(self, input_img, metas, x):
        x_shape = x.shape
        x_flat = x.reshape(-1, 3)
        input_dict = self._as_input_dict(input_img)
        sdf_feat, hand_pose_results, obj_pose_results = self.model(input_dict, None, metas, mode='test')
        obj_sample_subset = kinematic_embedding(
            self.obj_point_latent,
            self.recon_scale,
            x_flat,
            x_flat.shape[0],
            obj_pose_results,
            'obj',
        )
        obj_sample_subset = obj_sample_subset.to(self.device)
        obj_sample_subset = obj_sample_subset.reshape((-1, self.obj_point_latent))
        
        sdf_obj = decode_sdf(self._obj_sdf_head(), sdf_feat, obj_sample_subset, 'obj')
        sdf_obj = sdf_obj.reshape(*x_shape[:-1], 1)
        
        return sdf_obj

    def set_context(self, input_img, metas):
        self._cached_input_img = input_img
        self._cached_metas = metas
        input_dict = self._as_input_dict(input_img)
        with torch.no_grad():
            sdf_feat, _, obj_pose_results = self.model(input_dict, None, metas, mode='test')
        self._cached_sdf_feat = sdf_feat.detach()
        self._cached_obj_pose_results = {
            k: (v.detach() if isinstance(v, torch.Tensor) else v)
            for k, v in obj_pose_results.items()
        }

    def distance_with_context(self, x):
        if self._cached_sdf_feat is None or self._cached_obj_pose_results is None:
            if self._cached_input_img is None or self._cached_metas is None:
                raise RuntimeError("No cached context. Call set_context(input_img, metas) first.")
            self.set_context(self._cached_input_img, self._cached_metas)

        x_shape = x.shape
        x_flat = x.reshape(-1, 3)
        obj_sample_subset = kinematic_embedding(
            self.obj_point_latent,
            self.recon_scale,
            x_flat,
            x_flat.shape[0],
            self._cached_obj_pose_results,
            'obj',
        )
        obj_sample_subset = obj_sample_subset.to(self.device).reshape((-1, self.obj_point_latent))
        sdf_obj = decode_sdf(self._obj_sdf_head(), self._cached_sdf_feat, obj_sample_subset, 'obj')
        return sdf_obj.reshape(*x_shape[:-1], 1)

    def sdf_and_gradient(self, input_img, metas, x, create_graph=True):
        x_query = x
        if not x_query.requires_grad:
            x_query = x.clone().detach().requires_grad_(True)
        sdf_obj = self.distance(input_img, metas, x_query)
        grad = torch.autograd.grad(
            outputs=sdf_obj.sum(),
            inputs=x_query,
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=False,
        )[0]
        return sdf_obj, grad

    def sdf_and_gradient_with_context(self, x, create_graph=True):
        x_query = x
        if not x_query.requires_grad:
            x_query = x.clone().detach().requires_grad_(True)
        sdf_obj = self.distance_with_context(x_query)
        grad = torch.autograd.grad(
            outputs=sdf_obj.sum(),
            inputs=x_query,
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=False,
        )[0]
        return sdf_obj, grad
        
        
    def gradient(self, x, distance, retain_graph=False, create_graph=False, allow_unused=False):
        return torch.autograd.grad([distance.sum()], [x], retain_graph=retain_graph, create_graph=create_graph, allow_unused=allow_unused)[0]

        
        
        
