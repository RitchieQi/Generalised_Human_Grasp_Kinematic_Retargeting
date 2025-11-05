import pytorch3d as p3d
import pytorch_volumetric as pv
import numpy as np
import trimesh as tm
import torch
from optimisation_tmp.robot import human
from torch.autograd import Function
from optimisation_tmp.loss import FCLoss
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from typing import List
from plotly import graph_objects as go
from torch.nn.functional import relu
import optimisation_tmp.icp as icp
from sklearn.decomposition import PCA
import hdbscan
class SDFLayer(Function):
    @staticmethod
    def forward(ctx, query_points, query_function):
        """
        input:
        --------------------------------
        query_points: N x 3
        query_function: function

        output:
        --------------------------------
        sdf_val: N x 1
        sdf_grad: N x 3
        """
        sdf_val, sdf_grad = query_function(query_points)
        ctx.save_for_backward(sdf_grad, sdf_val)
        return sdf_val

    @staticmethod
    def backward(ctx, grad_output):

        grad,val = ctx.saved_tensors
        grad[val < 0] *= -1 
        #grad = grad*torch.abs(val)
        grad_input = grad #* grad_output.unsqueeze(-1)
        return grad_input, None

        
#pytorch_volumetric
class object_sdf():
    def __init__(self, 
                 contact_threshold = 0.000,
                 cosine_threshold = -0.3,
                 obj_mesh = None,
                 hand_mesh = None,
                 device = "cuda", 
                 robot_contact = None):
        self.device = device
        self.hand_model = human(device=device)
        self.loss = FCLoss(device=device)
        self.relu = torch.nn.ReLU()
        self.contact_threshold = contact_threshold
        self.cosine_threshold = cosine_threshold    
        self.robot_contact = robot_contact
        self.obj_vert = None
        self.obj_face = None
        self.obj_points = None
        self.obj_normals = None
        self.hand_contact = None
        self.contact_target = None
        if obj_mesh is not None and hand_mesh is not None:
            self.reset(obj_mesh, hand_mesh, robot_contact)
    
    def reset(self, 
              obj_mesh:List[torch.Tensor], 
              hand_mesh:List[torch.Tensor],
              joints:torch.Tensor):
        self.obj_vert = obj_mesh[0].to(self.device)
        self.hand_model.load_hand_mesh(mesh_v = hand_mesh[0], mesh_f = hand_mesh[1], joints = joints)
        self.pv_sdf = pv.MeshSDF(pv.MeshObjectFactory(mesh_name=None, preload_mesh=dict(vertices=obj_mesh[0].squeeze().cpu().numpy(), faces=obj_mesh[1].squeeze().cpu().numpy()))) #the backend open3d is not compatible with cuda
        self.contact_target = None
        self.hand_contact = None
        self.obj_face = obj_mesh[1].to(self.device)
        self.obj_mesh = tm.Trimesh(vertices=self.obj_vert.squeeze().cpu().detach().numpy(), faces=self.obj_face.squeeze().cpu().detach().numpy())
        pts, pts_face_index = tm.sample.sample_surface_even(self.obj_mesh, 1024)
        pts_normal = np.array([self.obj_mesh.face_normals[i] for i in pts_face_index])
        self.obj_points = torch.tensor(pts).float().to(self.device)
        self.obj_normals = torch.tensor(pts_normal).float().to(self.device)

    def hand_object_contact(self):
        """Need object mesh information first, which is not likely to be available in the real time.
        """
        #hand contact points: 5*n*3
        hand_contact_points = torch.tensor(self.hand_model.get_contact_zones()).float().to(self.device)
        #obj contact points: 5*N*3
        object_vert = self.obj_vert.unsqueeze(0).repeat(hand_contact_points.size(0),1,1)
        nn = p3d.ops.knn_points(object_vert, hand_contact_points, K = 1)
        nn_idx = nn.dists[..., 0].argmin(dim=1)
        #print(nn.dists.size())
        #print(object_vert[:,nn_idx,:].size())
        return self.obj_vert[nn_idx,:] 
        #print(nn.idx[..., 0].size())
        #return nn

    def hand_contact_cluster(self):
        """ We have predefined hand contact zones, but we need to figure out how many of them are contected with the object
        and where they are. Then we need to find out how many contact points could be provided by the robot.
        
        First, at least we need two antipadal points to form a grasp, so 2 clusters are needed.
        Followed by a bunch of nasty if-else statements.
        if contacts of robot hand is 2, no more cluster needed.
        if contacts of robot hand is greater than 2, and grester tham the contacts of human, just use the human contact points.
        if contacts of robot hand is greater than 2, and less than the contacts of human, further clustering is needed.

        at the end, all the contact points will be transformed back to the mano frame to determine the finger correspondance.
        """

        hand_contact_points = self.hand_model.get_contact_zones().to(torch.float32).to(self.device)
        sdf_query,sdf_normal = self.pv_sdf(hand_contact_points) #assume shape: B,N,1
        finger_normals = self.hand_model.get_contact_normals()

        mask = (sdf_query <= self.contact_threshold)
        finger_indices, point_indices = torch.nonzero(mask, as_tuple=True)
        finger_mask = finger_indices.unique()
        n_hand_contacts = finger_mask.size(0)
        #print('mano aware', n_hand_contacts)
        value, contact_mask= torch.min(sdf_query[finger_mask,:], dim=1)
        contact_pivot = hand_contact_points[finger_mask, contact_mask, :]
        contact_normal = finger_normals[finger_mask, contact_mask, :]
       
        kmeans = KMeans(n_clusters=2, random_state=0).fit(contact_normal.cpu().detach().numpy())
        label = torch.tensor(kmeans.labels_, dtype=torch.long).to(self.device)

        if self.robot_contact == 2:
            contact_cluster_0 = contact_pivot[label == 0].mean(dim=0)
            contact_cluster_1 = contact_pivot[label == 1].mean(dim=0)
            return torch.vstack([contact_cluster_0, contact_cluster_1])
        elif self.robot_contact > 2 and self.robot_contact > n_hand_contacts or self.robot_contact == n_hand_contacts:
            finger_1 = torch.argmax(torch.bincount(label))
            finger_0 = contact_pivot[label != finger_1].mean(dim=0)
            rest_finger = n_hand_contacts - 1
            kmeans_rest = KMeans(n_clusters=rest_finger, random_state=0).fit(contact_pivot[label == finger_1].cpu().detach().numpy())
            label_rest = kmeans_rest.labels_
            centriods = kmeans_rest.cluster_centers_
            return torch.vstack([finger_0, torch.tensor(centriods).to(self.device)])
        elif self.robot_contact > 2 and self.robot_contact < n_hand_contacts:
            finger_1 = torch.argmax(torch.bincount(label))
            finger_0 = contact_pivot[label != finger_1].mean(dim=0)
            rest_finger = self.robot_contact - 1
            kmeans_rest = KMeans(n_clusters=rest_finger, random_state=0).fit(contact_pivot[label == finger_1].cpu().detach().numpy())
            label_rest = kmeans_rest.labels_
            centriods = kmeans_rest.cluster_centers_
            return torch.vstack([finger_0, torch.tensor(centriods).to(self.device)])
    
    def hand_contact_points(self):
        contact_pack = self.hand_model.get_contact_points_normals_packed()
        sdf_val, sdf_normal = self.pv_sdf(contact_pack[..., :3])
        #mask = (sdf_val <= self.contact_threshold)
        point_indices = torch.nonzero(sdf_val <= self.contact_threshold, as_tuple=True)[0]
        contact_points = contact_pack[point_indices, :3]
        return contact_points        
    
    def hand_contact_normals(self):
        contact_pack = self.hand_model.get_contact_points_normals_packed()
        sdf_val, sdf_normal = self.pv_sdf(contact_pack[..., :3])
        #mask = (sdf_val <= self.contact_threshold)
        point_indices = torch.nonzero(sdf_val <= self.contact_threshold, as_tuple=True)[0]
        contact_normals = contact_pack[point_indices, 3:]
        return contact_normals
    
    def hand_contact_objnormals(self):
        contact_pack = self.hand_model.get_contact_points_normals_packed()
        sdf_val, sdf_normal = self.pv_sdf(contact_pack[..., :3])
        #mask = (sdf_val <= self.contact_threshold)
        point_indices = torch.nonzero(sdf_val <= self.contact_threshold, as_tuple=True)[0]
        contact_normals = sdf_normal[point_indices]
        return contact_normals
        
    def hand_contact_cluster_mano_agnostic(self):
        contact_pack = self.hand_model.get_contact_points_normals_packed()
        sdf_val, sdf_normal = self.pv_sdf(contact_pack[..., :3])
        hand_normal = contact_pack[..., 3:]
        #mask = (sdf_val <= self.contact_threshold)
        condition_sdf = sdf_val <= self.contact_threshold
        normal_similarity = torch.sum(hand_normal * sdf_normal, dim=-1)
        condition_antipodal = normal_similarity < -0.5

        point_indices = torch.nonzero(condition_sdf & condition_antipodal, as_tuple=True)[0]
        print("contact points number", len(point_indices))
        contact_points = contact_pack[point_indices, :3]
        contact_normals = sdf_normal[point_indices]
        # contact_normals = contact_pack[point_indices, 3:]
        #print(sdf_val,contact_points)
        
        #normalize the points
        # contact_points_ = contact_points - contact_points.mean(dim=0)
        # contact_points_n = contact_points_ / torch.norm(contact_points_, dim=-1).max()
        contact_points_n = contact_points
        
        # contact_normals_ = contact_normals - contact_normals.mean(dim=0)
        # contact_normals_n = contact_normals_ / torch.norm(contact_normals_, dim=-1).max()
        
        s_score = []
        for n in range(2, 6):
            kmeans = KMeans(n_clusters=n).fit(contact_points_n.cpu().detach().numpy())
            score = silhouette_samples(contact_points.cpu().detach().numpy(), kmeans.labels_, metric='l2')
            #s_score.append(process_silhouette_score(score, kmeans.labels_))
            s_score.append(score.mean())
        n_hand_contacts = torch.argmax(torch.tensor(s_score)) + 2
        print("silhouette score", s_score, n_hand_contacts)


        #contact_pack_n = torch.hstack([contact_points_n, contact_normals])
        contact_pack_n = contact_normals
        #contact_pack_n = contact_points_n
        kmeans_0 = KMeans(n_clusters=2).fit(contact_pack_n.cpu().detach().numpy())
        label = torch.tensor(kmeans_0.labels_, dtype=torch.long).to(self.device)
        label_unique = torch.unique(label)
        self.contact_0 = contact_points[label == 0]
        self.contact_1 = contact_points[label == 1]

        #find associated contact points
        # centers = kmeans_0.cluster_centers_
        # distances = torch.stack([torch.norm(contact_points_n - torch.tensor(centers[i]).to(self.device), dim=-1) for i in range(2)]).to(self.device)
        # min_indices = torch.argmin(distances, dim=1)
        # contact_points_0 = contact_points[min_indices[0]]
        # contact_points_1 = contact_points[min_indices[1]]
        #print("contact points", contact_points_0, contact_points_1)
        #print("dim check", distances.size(), min_indices.size(), contact_points.size(), contact_points_0.size(), contact_points_1.size())


        self.hand_contact = n_hand_contacts
        if self.robot_contact == 2:
            contact_cluster_0 = contact_points[label == 0].mean(dim=0)
            distance_0 = torch.norm(contact_points[label == 0] - contact_cluster_0, dim=-1)
            min_idx_0 = torch.argmin(distance_0)
            contact_points_0 = contact_points[label == 0][min_idx_0]
            contact_cluster_1 = contact_points[label == 1].mean(dim=0)
            distance_1 = torch.norm(contact_points[label == 1] - contact_cluster_1, dim=-1)
            min_idx_1 = torch.argmin(distance_1)
            contact_points_1 = contact_points[label == 1][min_idx_1]
            self.contact_target = torch.vstack([contact_points_0, contact_points_1])
        
        elif self.robot_contact > 2 and self.robot_contact > n_hand_contacts or self.robot_contact == n_hand_contacts:
            finger_1 = torch.argmax(torch.bincount(label))
            #print("finger_1", finger_1, label)
            finger_0 = contact_points[label != finger_1].mean(dim=0)
            distance_0 = torch.norm(contact_points[label != finger_1] - finger_0, dim=-1)
            min_idx_0 = torch.argmin(distance_0)
            finger_0_ = contact_points[label != finger_1][min_idx_0]
            #finger_0 = contact_points[min_indices[label_unique != finger_1]]
            rest_finger = n_hand_contacts - 1
            kmeans_rest = KMeans(n_clusters=rest_finger, random_state=0).fit(contact_points[label == finger_1].cpu().detach().numpy())
            label_rest = kmeans_rest.labels_
            centroids = kmeans_rest.cluster_centers_
            distance_rest = torch.stack([torch.norm(contact_points[label == finger_1] - torch.tensor(centroids[i]).to(self.device), dim=-1) for i in range(rest_finger)]).to(self.device)

            min_indices = torch.argmin(distance_rest, dim=1)
            centroids_ = contact_points[label == finger_1][min_indices]
            #self.contact_target = torch.vstack([finger_0_, torch.tensor(centriods).to(self.device)])
            self.contact_target = torch.vstack([finger_0_, centroids_])
            
        elif self.robot_contact > 2 and self.robot_contact < n_hand_contacts:
            finger_1 = torch.argmax(torch.bincount(label))
            finger_0 = contact_points[label != finger_1].mean(dim=0)
            distance_0 = torch.norm(contact_points[label != finger_1] - finger_0, dim=-1)
            min_idx_0 = torch.argmin(distance_0)
            finger_0_ = contact_points[label != finger_1][min_idx_0]
            #finger_0 = contact_points[min_indices[label_unique != finger_1]]
            rest_finger = self.robot_contact - 1
            kmeans_rest = KMeans(n_clusters=rest_finger, random_state=0).fit(contact_points[label == finger_1].cpu().detach().numpy())
            label_rest = kmeans_rest.labels_
            centroids = kmeans_rest.cluster_centers_
            distance_rest = torch.stack([torch.norm(contact_points[label == finger_1] - torch.tensor(centroids[i]).to(self.device), dim=-1) for i in range(rest_finger)]).to(self.device)

            min_indices = torch.argmin(distance_rest, dim=1)
            centroids_ = contact_points[label == finger_1][min_indices]
            self.contact_target = torch.vstack([finger_0_, centroids_])
        print("contact target", self.contact_target, self.contact_target.size())
        
    def hand_contact_cluster_v2(self):
        def reorder_indices(ides):
            print("ides", ides)
            length = ides.size(0)
            sorted, ids = torch.sort(ides)
            # print("indices inplace", ids)
            # print("indices sorted", sorted)
            # reverse_indices = torch.empty_like(ids).to(self.device)
            # reverse_indices[ids] = torch.arange(length).to(self.device)
            return ids
            
        contact_pack = self.hand_model.get_contact_points_normals_packed()
        fingertip = self.hand_model.get_fingertips()
        sdf_val, sdf_normal = self.pv_sdf(contact_pack[..., :3])
        hand_points, hand_normals = contact_pack[..., :3], contact_pack[..., 3:]
        
        condition_sdf = sdf_val <= self.contact_threshold
        normal_similarity = torch.sum(hand_normals * sdf_normal, dim=-1)
        condition_antipodal = normal_similarity < self.cosine_threshold
        point_indices = torch.nonzero(condition_sdf & condition_antipodal, as_tuple=True)[0]
        print("contact points number", len(point_indices))
        contact_points = hand_points[point_indices]
        contact_normals = sdf_normal[point_indices]
        
        #normalize the points
        contact_points_c = contact_points - contact_points.mean(dim=0)
        contact_points_n = contact_points_c / torch.norm(contact_points_c, dim=-1).max()
        #reconcatenate the points and normals
        contact_pack_n = torch.hstack([contact_points_n, contact_normals])
        
        #cluster the contact normals first
        #TODO: why not cluster at the same time? Both points and normals?
        kmeans_n = KMeans(n_clusters=2).fit(contact_pack_n.cpu().detach().numpy())
        label_n = torch.tensor(kmeans_n.labels_, dtype=torch.long).to(self.device)
        label_n_unique = torch.unique(label_n)
        contact_0 = contact_points_n[label_n == 0]
        contact_1 = contact_points_n[label_n == 1]
        self.contact_0 = contact_points[label_n == 0]
        self.contact_1 = contact_points[label_n == 1]
        self.contact_0_normal = contact_normals[label_n == 0]
        self.contact_1_normal = contact_normals[label_n == 1]
        #find the group with more contact points
        #contact_group_ = torch.argmax(torch.tensor([contact_group_0.size(0), contact_group_1.size(0)]))
        contact_group_ = contact_0 if contact_0.size(0) > contact_1.size(0) else contact_1
        contact_group_0 = contact_0 if contact_0.size(0) < contact_1.size(0) else contact_1
        
        contact_normal_ = self.contact_0_normal if contact_0.size(0) > contact_1.size(0) else self.contact_1_normal
        contact_group_o = self.contact_0 if contact_0.size(0) > contact_1.size(0) else self.contact_1
        contact_position_0 = torch.mean(contact_group_0, dim=0) 
        
        s_score = []
        cluster = []
        for n in range(2, 5):
            kmeans = KMeans(n_clusters=n).fit(contact_group_.cpu().detach().numpy())
            score = silhouette_samples(contact_group_.cpu().detach().numpy(), kmeans.labels_, metric='l2')
            s_score.append(score.mean())
            cluster.append(kmeans)
        highest_score = torch.max(torch.tensor(s_score))
        
        if highest_score < 0.65:
            n_hand_contacts = 2            
        else:
            highest_idx = torch.argmax(torch.tensor(s_score))
            n_hand_contacts = highest_idx + 3
            #highest_cluster = cluster[highest_idx]
        
        if self.robot_contact == 2:
            self.contact_target = torch.vstack([torch.mean(contact_group_, dim=0), contact_position_0])
        elif n_hand_contacts == 2:
            self.contact_target = torch.vstack([torch.mean(contact_group_, dim=0), contact_position_0])
        elif self.robot_contact > 2 and n_hand_contacts > 2 and self.robot_contact > n_hand_contacts or self.robot_contact == n_hand_contacts:
            cluster_ = cluster[highest_idx]
            centriods = torch.tensor(cluster_.cluster_centers_).to(self.device)
            self.contact_target = torch.vstack([centriods, contact_position_0])
        elif self.robot_contact > 2 and n_hand_contacts > 2 and self.robot_contact < n_hand_contacts:
            cluster_idx = self.robot_contact - 3
            cluster_ = cluster[cluster_idx]
            centriods = torch.tensor(cluster_.cluster_centers_).to(self.device)
            self.contact_target = torch.vstack([centriods, contact_position_0])
        
        if self.robot_contact >2 and n_hand_contacts > 2:
            self.sub_contact = []
            self.sub_contact_normal = []
            clabel_n = torch.tensor(cluster_.labels_, dtype=torch.long).to(self.device)
            for clabel in torch.unique(clabel_n):
                self.sub_contact.append(contact_group_o[clabel_n == clabel])
                self.sub_contact_normal.append(contact_normal_[clabel_n == clabel])
                
        print("contact target", self.contact_target, self.contact_target.size())
        print("fingertip", fingertip, fingertip.size())
        self.contact_target = (self.contact_target * torch.norm(contact_points_c, dim=-1).max()) + contact_points.mean(dim=0)
        indices = icp.nn_reg(self.contact_target.cpu().detach().numpy(), fingertip.squeeze().cpu().detach().numpy())
        indices_2 = icp.nn_reg(fingertip.squeeze().cpu().detach().numpy(), self.contact_target.cpu().detach().numpy())
        print("indices", indices)
        print("indices_2", indices_2)
        # print("indices", indices)
        # print("contact target before", self.contact_target, self.contact_target.size())
        # indices = torch.tensor(indices_2, dtype=torch.long).to(self.device)
        # reverse_indices = torch.empty_like(indices).to(self.device)
        # print("reverse_indices", reverse_indices)
        # reverse_indices[indices] = torch.arange(indices.size(0)).to(self.device)
        # print("reverse_indices", reverse_indices)
        # test_indices = reorder_indices(indices)
        # print("test_indices", test_indices)
        indices = torch.tensor(indices_2, dtype=torch.long).to(self.device)
        reverse_indices = reorder_indices(indices)
        self.contact_target = self.contact_target[reverse_indices]
        # print("contact target", self.contact_target, self.contact_target.size())
        # print("joint", fingertip)

    def hand_contact_cluster_hdbscan(self):

        def filter_labels_by_threshold(labels, indices, threshold=4):
            """
            Filters out labels that appear less than `threshold` times and removes the corresponding elements in indices.

            Parameters:
                labels (np.ndarray): Array of labels.
                indices (np.ndarray): Array of corresponding indices.
                threshold (int): Minimum occurrences required to keep a label.

            Returns:
                np.ndarray: Filtered labels.
                np.ndarray: Filtered indices.
            """
            unique_labels, counts = np.unique(labels, return_counts=True)
            if len(unique_labels) == 1:
                return labels, indices  # No filtering needed
            valid_labels = unique_labels[counts >= threshold]
            mask = np.isin(labels, valid_labels)
            return labels[mask], indices[mask]

        def find_n_section_pca(points, n):
            if isinstance(points, list) or isinstance(points, np.ndarray):
                try:
                    points = np.vstack(points)  # Stack into a single (N, D) array
                except ValueError:
                    print("Error: Arrays in pca_base have inconsistent shapes. Fixing...")
                    points = np.concatenate([p for p in points if p.shape[1] == 3])
            # Compute mean of the dataset (center of mass)
            pca = PCA(n_components=1)
            projected = pca.fit_transform(points)  # Project points onto 1D principal axis

            # Find n-section points along the principal component
            min_proj, max_proj = projected.min(), projected.max()
            section_values = np.linspace(min_proj, max_proj, num=n+1)[1:-1]  # Exclude endpoints

            # Ensure correct shape for PCA inverse transform
            section_values = section_values.reshape(-1, 1)

            # Convert back to original space using PCA inverse transform
            section_points = pca.inverse_transform(section_values)

            return section_points

        contact_pack = self.hand_model.get_contact_points_normals_packed()
        sdf_val, sdf_normal = self.pv_sdf(contact_pack[..., :3])
        hand_points, hand_normals = contact_pack[..., :3], contact_pack[..., 3:]
        
        condition_sdf = sdf_val <= self.contact_threshold
        normal_similarity = torch.sum(hand_normals * sdf_normal, dim=-1)
        condition_antipodal = normal_similarity < self.cosine_threshold
        point_indices = torch.nonzero(condition_sdf & condition_antipodal, as_tuple=True)[0]
        # print("contact points number", len(point_indices))
        contact_points = hand_points[point_indices]
        contact_normals = sdf_normal[point_indices]

        #normalize the points
        contact_points_c = contact_points - contact_points.mean(dim=0)
        contact_normals = contact_normals.cpu().detach().numpy()    
        contact_points  = contact_points.cpu().detach().numpy()

        #cluster the contact normals first
        clusterer_n = hdbscan.HDBSCAN(min_cluster_size=10)
        clusterer_n.fit(contact_normals)

        labels_n = np.array(clusterer_n.labels_)
        clusterer_p = hdbscan.HDBSCAN(min_cluster_size=4)
        clusterer_p.fit(contact_points)

        labels_p = np.array(clusterer_p.labels_)
        # print("labels_n", labels_n)
        # print("labels_p", labels_p)

        #TODO: be addaptive, rise the min_cluster_size if the number of clusters more than 5

        
        # filer out -1
        valid_mask =  (labels_n != -1) & (labels_p != -1)
        contact_points = contact_points[valid_mask]
        contact_normals = contact_normals[valid_mask]
        labels_n = labels_n[valid_mask]
        labels_p = labels_p[valid_mask]

        n_normal_components = np.unique(labels_n).size 
        uniq_labels_n = np.unique(labels_n) 
        n_hand_contacts = np.unique(labels_p).size
        uniq_labels_p = np.unique(labels_p) 

        self.labels_n = labels_n
        self.labels_p = labels_p
        self.contact_points = contact_points
        self.contact_normals = contact_normals

        if np.all(labels_n==-1) or np.all(labels_p==-1):
            print("hdbscan failed, try kmeans")
            self.hand_contact_cluster_v2()
            return 0
        centroid_n = {}
        centroid_p = {}
        n_components = {}
        p_collection = {}
        for labels in uniq_labels_n:
            indices = np.where(labels_n == labels)[0]
            p_in_n = labels_p[indices]
            p_in_n, indices = filter_labels_by_threshold(p_in_n, indices)
            n_components[labels] = p_in_n
            centroid_n[labels] = np.mean(contact_normals[indices], axis=0)/np.linalg.norm(contact_normals[indices])
            p_cenroid_ = []
            p_collector = []
            for p in np.unique(p_in_n):
                indices_p = np.where(p_in_n == p)[0]
                p_cenroid_.append(np.mean(contact_points[indices[indices_p]], axis=0))
                p_collector.append(contact_points[indices[indices_p]])
            centroid_p[labels] = np.array(p_cenroid_)
            p_collection[labels] = np.array(p_collector)

        self.n_components = n_components
        centroid_n_ = np.array([centroid_n[labels] for labels in uniq_labels_n]) # the normal cluster centroids array
        centroid_p_ = np.vstack([centroid_p[labels] for labels in uniq_labels_n]) # n np.array n is the number of normal clusters
        centroid_p_n = np.vstack([centroid_p[labels].mean(axis=0) for labels in uniq_labels_n]) 


        if n_normal_components > 2:
            dist_matrix = cdist(centroid_n_, centroid_n_)
            i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
            contact_0 = centroid_p[i].mean(axis=0) if centroid_p[i].shape[0] <= centroid_p[j].shape[0] else centroid_p[j].mean(axis=0)
            contact_1 = centroid_p[j].mean(axis=0) if centroid_p[i].shape[0] <= centroid_p[j].shape[0] else centroid_p[i].mean(axis=0)
            pca_base_ = p_collection[i] if centroid_p[i].shape[0] > centroid_p[j].shape[0] else p_collection[j]
            #pca_base = pca_base.squeeze()
            pca_base = np.vstack(np.atleast_1d(pca_base_)).astype(np.float32)
            uniq_labels_n = np.delete(uniq_labels_n, np.where(uniq_labels_n == i))
            uniq_labels_n = np.delete(uniq_labels_n, np.where(uniq_labels_n == j))
            pca_base_list = np.vstack([p for i in uniq_labels_n for p in p_collection[i]])
            # print("pca_base_list", pca_base_list.shape)
            pca_base = np.vstack([pca_base, pca_base_list])


            # print("uniq_labels_n", uniq_labels_n)

        else:
            contact_0 = centroid_p[0].mean(axis=0) if centroid_p[0].shape[0] <= centroid_p[1].shape[0] else centroid_p[1].mean(axis=0)
            contact_1 = centroid_p[1].mean(axis=0) if centroid_p[0].shape[0] <= centroid_p[1].shape[0] else centroid_p[0].mean(axis=0)
            uniq_labels_n = np.delete(uniq_labels_n, np.where(uniq_labels_n == 0))
            uniq_labels_n = np.delete(uniq_labels_n, np.where(uniq_labels_n == 1))
            pca_base_ = p_collection[0] if centroid_p[0].shape[0] > centroid_p[1].shape[0] else p_collection[1]
            pca_base = np.vstack(np.atleast_1d(pca_base_)).astype(np.float32)

            
        if self.robot_contact == 2:
            self.contact_target = torch.tensor(np.vstack([contact_0, contact_1])).to(self.device)
        
        elif self.robot_contact > 2 and self.robot_contact <= n_normal_components:
            self.contact_target = np.vstack([contact_0, contact_1])
            for i in range(self.robot_contact - 2):
                idx = uniq_labels_n[i]
                self.contact_target = np.vstack([self.contact_target, centroid_p_n[idx]])
            self.contact_target = torch.tensor(self.contact_target).to(self.device)
        
        elif self.robot_contact > 2 and self.robot_contact >= n_hand_contacts:
            self.contact_target = torch.tensor(centroid_p_).to(self.device)
        
        elif self.robot_contact > 2 and self.robot_contact < n_hand_contacts and self.robot_contact > n_normal_components:
            # print("pca_base", pca_base)
            contact_2 = find_n_section_pca(pca_base, self.robot_contact)
            self.contact_target = torch.tensor(np.vstack([contact_0, contact_2]), dtype=torch.float32).to(self.device)


        print(self.contact_target)

    def sdf_loss(self, points):
        sdf_val = SDFLayer.apply(points, self.pv_sdf)
        return (sdf_val**2).sum()
            
    def extreme_distance(self, n, n_t):
        N,D = n.size()
        N_t,D_t = n_t.size()
        rank = []
        remaining = torch.range(0,N_t-1, dtype=torch.long)
        while len(rank) < N_t:
            for ax in range(D):
                if len(remaining) == 0:
                    break
                min_id = remaining[torch.argmin(n_t[remaining,ax])]
                max_id = remaining[torch.argmax(n_t[remaining,ax])]

                if torch.abs(n_t[min_id,ax]) > torch.abs(n_t[max_id,ax]):
                    rank.append(min_id)
                    remaining = remaining[remaining != min_id]
                else:
                    rank.append(max_id)
                    remaining = remaining[remaining != max_id]        

                if len(remaining) == 0:
                    break
        rank = torch.stack(rank)

        repeats = (N+N_t-1)//N_t
        rank = rank.repeat(repeats)[:N]
        target_n_t = n_t[rank]

        loss = torch.norm(n - target_n_t, dim=-1).sum()

        return  loss

    def minimum_internal_distance(self, x):
        x = x.view(-1, 3)
        dist_matrix = torch.cdist(x, x, p=2)
        eye = torch.eye(x.shape[0], device=x.device) * 1e6
        dist_matrix = dist_matrix + eye        
        min_dists = torch.min(dist_matrix, dim=1)[0]
        penalty = self.relu(0.01 - min_dists)
        return penalty.sum()

    def get_weighted_edges(self, x, w):
        _,normal = self.pv_sdf(x)
        f, we = self.loss.linearized_cone(normal, w)
        return we
    
    def loss_fc(self, x, w):
        # if self.contact_target is None:
        #     self.hand_contact_cluster_v2()
        val, normal = self.pv_sdf(x)
        
        x_norm = x/torch.norm(x, dim = -1).max()
        G = self.loss.x_to_G(x_norm)
        
        GG = self.loss.loss_8a(G)
        f, we = self.loss.linearized_cone(normal, w)
        Gf = self.loss.loss_8b(f, G)
        
        
        intFC = self.loss.inter_fc(w)
        
        sdf = self.sdf_loss(x)

        # print("check distance", (torch.norm(x.squeeze(0) - self.contact_target, dim=-1)))
        e_dist = self.relu((torch.norm(x.squeeze(0) - self.contact_target, dim=-1) - 0.01)).sum()
        
        minimum_internal_distance = self.minimum_internal_distance(x)
        sum_ = 1*sdf + 1*Gf + GG + intFC + 10*e_dist + 1*minimum_internal_distance
        return dict(sdf=sdf, Gf=Gf,  GG=GG, intFC=intFC, distance=e_dist, inter_dist = minimum_internal_distance, loss=sum_)
    
    def gf_residual(self, x, w):
        val, normal = self.pv_sdf(x)
        x_norm = x/torch.norm(x, dim = -1).max()
        G = self.loss.x_to_G(x_norm)
        GG = self.loss.loss_8a(G)
        f, we = self.loss.linearized_cone(normal, w)
        Gf = self.loss.loss_8b(f, G)
        intFC = self.loss.inter_fc(w)
        sum_ = Gf + intFC + GG 
        return dict(Gf=Gf, GG=GG, intFC=intFC, loss=sum_)

    def loss_gf(self, x, w):
        if self.contact_target is None:
            self.hand_contact_cluster_v2()
        val, normal = self.pv_sdf(x)
        
        x_norm = x/torch.norm(x, dim = -1).max()
        G = self.loss.x_to_G(x_norm)
        GG = self.loss.loss_8a(G)
        f, we = self.loss.linearized_cone(normal, w)
        Gf = self.loss.loss_8b(f, G)
        intFC = self.loss.inter_fc(w)
        sum_ = Gf + intFC + GG 
        return dict(Gf=Gf, GG=GG, intFC=intFC, loss=sum_)
    
    def loss_fc_sfw(self, x, w):
        if self.contact_target is None:
            self.hand_contact_cluster_v2()
        val, normal = self.pv_sdf(x)

        G = self.loss.x_to_G(x)
        GG = self.loss.loss_8a(G)
        f, we = self.loss.linearized_cone(normal, w)
        #wrench = self.loss.wrench_space(x, we)
        Gf = self.loss.loss_8b(f, G)
        intFC = self.loss.inter_fc(w)
        sdf = self.sdf_loss(x)
        #e_dist = self.extreme_distance(x.squeeze(0), self.hand_object_contact())
        e_dist = self.relu((torch.norm(x.squeeze(0) - self.contact_target, dim=-1) - 0.01)).sum()
        #print("sdf",sdf, "Gf", Gf, "GG", GG, "intFC", intFC, "e_dist", e_dist)
        sum_ = sdf + Gf + GG + 0 + 5*e_dist
        return dict(sdf=sdf, Gf=Gf,  GG=GG, distance=e_dist, loss=sum_)

    def wrench_hull(self, x, w, with_x = False):
        if self.contact_target is None:
            self.hand_contact_cluster_v2()
        def convex_hull(wrench):
            """
            """
            #print(wrench.size())
            w_np = wrench.view(-1, 6).cpu().detach().numpy()
            #print(np.linalg.matrix_rank(w_np))
            #print(w_np)
            w_hull = ConvexHull(w_np)
            eq = w_hull.equations
            coef = eq[:, :-1]
            b = eq[:, -1]
            
            dist = np.abs(b)/np.linalg.norm(coef, ord=2, axis=-1)
            # print("volume", w_hull.volume)  
            #print(dist)
            idx = np.argmin(dist)
            vertices = w_hull.simplices[idx]
            selected = wrench[:, vertices, :]
            return torch.norm(selected,p=2,dim=-1).mean(), torch.tensor(dist[idx]).to(wrench.device)
        B,N,d = x.shape
        val, normal = self.pv_sdf(x)
        x_norm = x
        f, we = self.loss.linearized_cone(normal, w) # f: B x N x 3, we: B x N x 4 x 3
        # calculate the wrench space
        w_space = self.loss.wrench_space(x_norm, we)
        selected, dist = convex_hull(w_space)
        intFC = self.loss.inter_fc(w)

        G_ = self.loss.x_to_G(x_norm)
        GG_ = self.loss.loss_8a(G_)
        gf = self.loss.loss_8b(f, G_)

        if with_x:
            
            sdf = self.sdf_loss(x)
            e_dist = self.relu((torch.norm(x.squeeze(0) - self.contact_target, dim=-1) - 0.05)).sum()
            G = self.loss.x_to_G(x)
            GG = self.loss.loss_8a(G)
            sum_ = 1*sdf + 1.*gf + 1*GG + 1*e_dist - 1*selected +1*intFC
            return dict(sdf=sdf, Gf=gf, GG=GG, distance=e_dist, intFC = intFC, radius=dist, loss=sum_)
        else:
            sum_ = gf + intFC - 1*selected
            return dict(Gf=gf, intFC=intFC, radius=dist, loss=sum_)

    def draw(self, color="green", opacity = 0.5):
        obj_vert = self.obj_vert.squeeze().cpu().detach().numpy()
        obj_face = self.obj_face.squeeze().cpu().detach().numpy()
        mesh_obj = go.Mesh3d(x=obj_vert[:,0], y=obj_vert[:,1], z=obj_vert[:,2], i=obj_face[:,0], j=obj_face[:,1], k=obj_face[:,2], color=color, opacity=opacity)
        return mesh_obj
    
    def collision_check(self, points):
        sdf_val, sdf_normal = self.pv_sdf(points.squeeze(0))
        #print(sdf_val[sdf_val <= 0].size())
        # penetration = sdf_val[sdf_val <= 0].mean()
        # print(penetration)
        return relu(-sdf_val).sum()