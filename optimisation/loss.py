import numpy as np
import torch
import utils as mb

class FCLoss:
    def __init__(self, device='cuda'):

        self.device = device
        
        self.transformation_matrix = torch.tensor(np.array([[0,0,0,0,0,-1,0,1,0], [0,0,1,0,0,0,-1,0,0], [0,-1,0,1,0,0,0,0,0]])).float().to(device)
        self.eye3 = torch.tensor(np.eye(3).reshape(1, 1, 3, 3)).float().to(device)
        self.eye6 = torch.tensor(np.eye(6).reshape(1,6,6)).float().to(device)

        self.eps = torch.tensor(0.01).float().to(device)
        self.mu = torch.tensor(0.1).float().to(device)
        self.z = torch.tensor([0., 0., 1.]).float().view(1, 1, 3).to(self.device)
        self.e1 = torch.tensor([self.mu, 0, 1]).float().view(1, 1, 3).to(self.device)
        self.e2 = torch.tensor([0, self.mu, 1]).float().view(1, 1, 3).to(self.device)
        self.e3 = torch.tensor([-self.mu, 0, 1]).float().view(1, 1, 3).to(self.device)
        self.e4 = torch.tensor([0, -self.mu, 1]).float().view(1, 1, 3).to(self.device)
        
        
        self.relu = torch.nn.ReLU()


    def l2_norm(self, x):
        if len(x.shape) == 3:
            return torch.sum(x*x, (1, 2))
        if len(x.shape) == 2:
            return torch.sum(x*x, (1))
        raise ValueError

    def x_to_G(self, x):
        """
        x: B x N x 3
        G: B x 6 x 3N
        """
        B = x.shape[0]
        N = x.shape[1]
        xi_cross = torch.matmul(x, self.transformation_matrix).reshape([B,N,3,3]).transpose(1, 2).reshape([B, 3, 3*N])
        I = self.eye3.repeat([B, N, 1, 1]).transpose(1,2).reshape([B, 3, 3*N])
        G = torch.stack([I, xi_cross], 1).reshape([B, 6, 3*N])
        return G
    
    def loss_8a(self, G):
        """
        G: B x 6 x 3N
        """
        Gt = G.transpose(1,2)
        temp = self.eps * self.eye6
        temp = torch.matmul(G, Gt) - temp
        #eigval = torch.symeig(temp.cpu(), eigenvectors=True)[0].to(self.device)
        eigval = torch.linalg.eigvalsh(temp, UPLO='U').to(self.device)
        rnev = self.relu(-eigval)
        result = torch.sum(rnev * rnev, 1)
        return result
    
    def loss_8b(self, f, G): 
        """
        G: B x 6 x 3N
        f: B x N x 3
        """
        B = f.shape[0]
        N = f.shape[1]
        return self.relu(self.l2_norm(torch.matmul(G, f.reshape(B, 3*N, 1))))
    
    def inter_fc(self, w):
        # w: B x N x 4
        B, N, d = w.shape
        # do we need softmax here?
        # w = torch.nn.functional.softmax(w, dim=-1)
        
        inter = self.relu(-w).sum()
        ones = torch.ones([B, N]).to(self.device)

        #sum_N = torch.mean((torch.sum(w, dim=-1) - ones)**2)
        sum_N = torch.sum(self.relu(-(ones - torch.sum(w, dim=-1))))
        #sum_B = self.relu((torch.sum(w) - B)**2)

        return inter + sum_N #+ sum_B

    def linearized_cone(self, normal, w):
        """
        input:
        --------------------------------
        normal: B x N x 3
        f: B x N x 3

        output:
        --------------------------------
        w: B x N x 4

        Return the weights of the 4-edge friction pyramid
        """
        B = normal.shape[0]
        N = normal.shape[1]
        # construct 4-edge friction pyramid 
        z = self.z.repeat([B, N, 1])
        e1 = self.e1.repeat([B, N, 1])
        e2 = self.e2.repeat([B, N, 1])
        e3 = self.e3.repeat([B, N, 1])
        e4 = self.e4.repeat([B, N, 1])

        # f = w_0e_0 + , w > 0
        # fe^-1 > 0
        E = torch.stack([e1, e2, e3, e4], -2) # B x N x 4 x 3
        T = mb.multi_rotation_vectors_batch(z, normal) # B x N x 3 x 3

        #apple T to E
        E_transed = torch.matmul(T, E.transpose(-2,-1)).transpose(-2,-1) # B x N x 4 x 3
        f = torch.matmul(w.unsqueeze(-2), E_transed) #contact force B x N x 3
        w_edges = (E_transed * w.unsqueeze(-1)).reshape(B, N, 4, 3)

        
        #print("f:",f, "w:",w, "FC:", E_transed,"edge force:", w_edges)
        return f, w_edges
    
    def dist_loss(self, x):
        d, normal = self.object_sdf.query_sdf(x)
        d = d.sum()

        return d * d, normal
    
    def wrench_space(self,x,we):
        B,N,_ = x.shape
        we = we.reshape(B, N*4, 3)
        #x_expanded = x.unsqueeze(-2).repeat(1,1,4,1)
        x_expanded = x.repeat(1,4,1)
        torque = torch.cross(x_expanded, we, dim=-1)
        wrench = torch.cat([we, torque], dim=-1)
        
        return wrench

    
    def loss_G(self, x,f):
        G = self.x_to_G(x)
        l8a = self.loss_8a(G)
        l8b = self.loss_8b(f, G)

        # l8c = self.loss_8c(normal)
        # l8d = self.dist_loss(obj_code, x)
        return l8a, l8b


    