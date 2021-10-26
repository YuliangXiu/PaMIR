"""
This file contains the definition of the SMPL model
"""
from __future__ import division, print_function

import torch
import os
import torch.nn as nn
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle


from pyrender import PerspectiveCamera, \
    DirectionalLight, SpotLight, PointLight, \
    MetallicRoughnessMaterial, \
    Primitive, Mesh, Node, Scene, \
    Viewer, OffscreenRenderer

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """ 
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat  


def rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat2mat(quat)

class TetraSMPLModel():
    def __init__(self, model_path, model_addition_path, age='adult', v_template=None):
        """
        SMPL model.
    
        Parameter:
        ---------
        model_path: Path to the SMPL model parameters, pre-processed by
        `preprocess.py`.
    
        """
        with open(model_path, 'rb') as f:
            params = pickle.load(f, encoding='latin1')

            self.J_regressor = params['J_regressor']
            self.weights = np.asarray(params['weights'])
            self.posedirs = np.asarray(params['posedirs'])
            
            if v_template is not None:
                self.v_template = v_template
            else:
                self.v_template = np.asarray(params['v_template'])
                
            self.shapedirs = np.asarray(params['shapedirs'])
            self.faces = np.asarray(params['f'])
            self.kintree_table = np.asarray(params['kintree_table'])
        
        params_added = np.load(model_addition_path)
        self.v_template_added = params_added['v_template_added']
        self.weights_added = params_added['weights_added']
        self.shapedirs_added = params_added['shapedirs_added']
        self.posedirs_added = params_added['posedirs_added']
        self.tetrahedrons = params_added['tetrahedrons']
        
        # import pdb; pdb.set_trace()

        id_to_col = {
            self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
        }
        self.parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }

        self.pose_shape = [24, 3]
        self.beta_shape = [10]
        self.trans_shape = [3]
        
        if age == 'kid':    
            v_template_smil = np.load(os.path.join(os.path.dirname(model_path), "smpl_kid_template.npy"))
            v_template_smil -= np.mean(v_template_smil, axis=0)
            v_template_diff = np.expand_dims(v_template_smil - self.v_template, axis=2)
            self.shapedirs = np.concatenate((self.shapedirs[:, :, :self.beta_shape[0]], v_template_diff), axis=2)
            self.beta_shape[0] += 1

        self.pose = np.zeros(self.pose_shape)
        self.beta = np.zeros(self.beta_shape)
        self.trans = np.zeros(self.trans_shape)

        self.verts = None
        self.verts_added = None
        self.J = None
        self.R = None
        self.G = None

        self.update()

    def set_params(self, pose=None, beta=None, trans=None):
        """
        Set pose, shape, and/or translation parameters of SMPL model. Verices of the
        model will be updated and returned.
    
        Prameters:
        ---------
        pose: Also known as 'theta', a [24,3] matrix indicating child joint rotation
        relative to parent joint. For root joint it's global orientation.
        Represented in a axis-angle format.
    
        beta: Parameter for model shape. A vector of shape [10]. Coefficients for
        PCA component. Only 10 components were released by MPI.
    
        trans: Global translation of shape [3].
    
        Return:
        ------
        Updated vertices.
    
        """
        if pose is not None:
            self.pose = pose
        if beta is not None:
            self.beta = beta
        if trans is not None:
            self.trans = trans
        self.update()
        return self.verts

    def update(self):
        """
        Called automatically when parameters are updated.
    
        """
        # how beta affect body shape
        v_shaped = self.shapedirs.dot(self.beta) + self.v_template
        v_shaped_added = self.shapedirs_added.dot(self.beta) + self.v_template_added
        # joints location
        self.J = self.J_regressor.dot(v_shaped)
        pose_cube = self.pose.reshape((-1, 1, 3))
        # rotation matrix for each joint
        self.R = self.rodrigues(pose_cube)
        I_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            (self.R.shape[0] - 1, 3, 3)
        )
        lrotmin = (self.R[1:] - I_cube).ravel()
        # how pose affect body shape in zero pose
        v_posed = v_shaped + self.posedirs.dot(lrotmin)
        v_posed_added = v_shaped_added + self.posedirs_added.dot(lrotmin)
        # world transformation of each joint
        G = np.empty((self.kintree_table.shape[1], 4, 4))
        G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
        for i in range(1, self.kintree_table.shape[1]):
            G[i] = G[self.parent[i]].dot(
                self.with_zeros(
                    np.hstack(
                        [self.R[i], ((self.J[i, :] - self.J[self.parent[i], :]).reshape([3, 1]))]
                    )
                )
            )
        # remove the transformation due to the rest pose
        G = G - self.pack(
            np.matmul(
                G,
                np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])
            )
        )
        self.G = G
        # transformation of each vertex
        T = np.tensordot(self.weights, G, axes=[[1], [0]])
        rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
        v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
        self.verts = v + self.trans.reshape([1, 3])
        T_added = np.tensordot(self.weights_added, G, axes=[[1], [0]])
        rest_shape_added_h = np.hstack((v_posed_added, np.ones([v_posed_added.shape[0], 1])))
        v_added = np.matmul(T_added, rest_shape_added_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
        self.verts_added = v_added + self.trans.reshape([1, 3])


    def rodrigues(self, r):
        """
        Rodrigues' rotation formula that turns axis-angle vector into rotation
        matrix in a batch-ed manner.
    
        Parameter:
        ----------
        r: Axis-angle rotation vector of shape [batch_size, 1, 3].
    
        Return:
        -------
        Rotation matrix of shape [batch_size, 3, 3].
    
        """
        theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
        # avoid zero divide
        theta = np.maximum(theta, np.finfo(np.float64).tiny)
        r_hat = r / theta
        cos = np.cos(theta)
        z_stick = np.zeros(theta.shape[0])
        m = np.dstack([
            z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
            r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
            -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
        ).reshape([-1, 3, 3])
        i_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            [theta.shape[0], 3, 3]
        )
        A = np.transpose(r_hat, axes=[0, 2, 1])
        B = r_hat
        dot = np.matmul(A, B)
        R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
        return R

    def with_zeros(self, x):
        """
        Append a [0, 0, 0, 1] vector to a [3, 4] matrix.
    
        Parameter:
        ---------
        x: Matrix to be appended.
    
        Return:
        ------
        Matrix after appending of shape [4,4]
    
        """
        return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

    def pack(self, x):
        """
        Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
        manner.
    
        Parameter:
        ----------
        x: Matrices to be appended of shape [batch_size, 4, 1]
    
        Return:
        ------
        Matrix of shape [batch_size, 4, 4] after appending.
    
        """
        return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

    def save_mesh_to_obj(self, path):
        """
        Save the SMPL model into .obj file.
    
        Parameter:
        ---------
        path: Path to save.
    
        """
        with open(path, 'w') as fp:
            for v in self.verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
        
    def save_tetrahedron_to_obj(self, path):
        """
        Save the tetrahedron SMPL model into .obj file.
    
        Parameter:
        ---------
        path: Path to save.
    
        """
        
        with open(path, 'w') as fp:
            for v in self.verts:
                fp.write('v %f %f %f 1 0 0\n' % (v[0], v[1], v[2]))
            for va in self.verts_added:
                fp.write('v %f %f %f 0 0 1\n' % (va[0], va[1], va[2]))
            for t in self.tetrahedrons + 1:
                fp.write('f %d %d %d\n' % (t[0], t[2], t[1]))
                fp.write('f %d %d %d\n' % (t[0], t[3], t[2]))
                fp.write('f %d %d %d\n' % (t[0], t[1], t[3]))
                fp.write('f %d %d %d\n' % (t[1], t[2], t[3]))


class TetraSMPL(nn.Module):
    """
    Implementation of tetrahedral SMPL model
    Modified from https://github.com/nkolot/GraphCMR/blob/master/models/smpl.py 
    """
    def __init__(self, model_file, model_additional_file, v_template=None):
        super(TetraSMPL, self).__init__()
        with open(model_file, 'rb') as f:
            smpl_model = pickle.load(f, encoding='iso-8859-1')
        smpl_model_addition = np.load(model_additional_file)    # addition parameters for tetrahedrons
        smpl_model.update(smpl_model_addition)
        if v_template is not None:
            smpl_model.update({'v_template': v_template})
        self.orig_vert_num = smpl_model['v_template'].shape[0]
        self.added_vert_num = smpl_model['v_template_added'].shape[0]
        self.total_vert_num = self.orig_vert_num + self.added_vert_num

        J_regressor = smpl_model['J_regressor'].tocoo()
        row = J_regressor.row
        col = J_regressor.col
        data = J_regressor.data
        i = torch.LongTensor([row, col])
        v = torch.FloatTensor(data)
        J_regressor_shape = [24, self.orig_vert_num + self.added_vert_num]

        smpl_model['weights'] = np.concatenate([smpl_model['weights'], smpl_model['weights_added']], axis=0)
        smpl_model['posedirs'] = np.concatenate([smpl_model['posedirs'], smpl_model['posedirs_added']], axis=0)
        smpl_model['shapedirs'] = np.concatenate([smpl_model['shapedirs'], smpl_model['shapedirs_added']], axis=0)
        smpl_model['v_template'] = np.concatenate([smpl_model['v_template'], smpl_model['v_template_added']], axis=0)
        self.register_buffer('J_regressor', torch.sparse.FloatTensor(i, v, J_regressor_shape).to_dense())
        self.register_buffer('weights', torch.FloatTensor(smpl_model['weights']))
        self.register_buffer('posedirs', torch.FloatTensor(smpl_model['posedirs']))
        self.register_buffer('v_template', torch.FloatTensor(smpl_model['v_template']))
        self.register_buffer('shapedirs', torch.FloatTensor(np.array(smpl_model['shapedirs'])))
        self.register_buffer('faces', torch.from_numpy(smpl_model['f'].astype(np.int64)))
        self.register_buffer('tetrahedrons', torch.from_numpy(smpl_model['tetrahedrons'].astype(np.int64)))
        self.register_buffer('kintree_table', torch.from_numpy(smpl_model['kintree_table'].astype(np.int64)))
        id_to_col = {self.kintree_table[1, i].item(): i for i in range(self.kintree_table.shape[1])}
        self.register_buffer('parent', torch.LongTensor([id_to_col[self.kintree_table[0, it].item()] for it in range(1, self.kintree_table.shape[1])]))

        self.pose_shape = [24, 3]
        self.beta_shape = [300]
        self.translation_shape = [3]

        self.pose = torch.zeros(self.pose_shape)
        self.beta = torch.zeros(self.beta_shape)
        self.translation = torch.zeros(self.translation_shape)

        self.verts = None
        self.J = None
        self.R = None
        
    def forward(self, pose, beta=None):
        device = pose.device
        batch_size = pose.shape[0]
        v_template = self.v_template[None, :]
        shapedirs = self.shapedirs.view(-1,300)[None, :].expand(batch_size, -1, -1)
        if beta is None:
            beta = self.beta[None,...].type_as(pose)
        beta = beta[:, :, None]
        v_shaped = torch.matmul(shapedirs, beta).view(batch_size, -1, 3) + v_template
        # batched sparse matmul not supported in pytorch
        J = []
        for i in range(batch_size):
            J.append(torch.matmul(self.J_regressor, v_shaped[i]))
        J = torch.stack(J, dim=0)
        # input it rotmat: (bs,24,3,3)
        if pose.ndimension() == 4:
            R = pose
        # input it rotmat: (bs,72)
        elif pose.ndimension() == 2:
            pose_cube = pose.view(-1, 3) # (batch_size * 24, 1, 3)
            R = rodrigues(pose_cube).view(batch_size, 24, 3, 3)
            R = R.view(batch_size, 24, 3, 3)
        I_cube = torch.eye(3)[None, None, :].to(device)
        # I_cube = torch.eye(3)[None, None, :].expand(theta.shape[0], R.shape[1]-1, -1, -1)
        lrotmin = (R[:,1:,:] - I_cube).view(batch_size, -1)
        posedirs = self.posedirs.view(-1,207)[None, :].expand(batch_size, -1, -1)
        v_posed = v_shaped + torch.matmul(posedirs, lrotmin[:, :, None]).view(batch_size, -1, 3)
        J_ = J.clone()
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parent, :]
        G_ = torch.cat([R, J_[:, :, :, None]], dim=-1)
        pad_row = torch.FloatTensor([0,0,0,1]).to(device).view(1,1,1,4).expand(batch_size, 24, -1, -1)
        G_ = torch.cat([G_, pad_row], dim=2)
        G = [G_[:, 0].clone()]
        for i in range(1, 24):
            G.append(torch.matmul(G[self.parent[i-1]], G_[:, i, :, :]))
        G = torch.stack(G, dim=1)

        rest = torch.cat([J, torch.zeros(batch_size, 24, 1).to(device)], dim=2).view(batch_size, 24, 4, 1)
        zeros = torch.zeros(batch_size, 24, 4, 3).to(device)
        rest = torch.cat([zeros, rest], dim=-1)
        rest = torch.matmul(G, rest)
        G = G - rest
        T = torch.matmul(self.weights, G.permute(1,0,2,3).contiguous().view(24,-1)).view(-1, batch_size, 4, 4).transpose(0,1)
        rest_shape_h = torch.cat([v_posed, torch.ones_like(v_posed)[:, :, [0]]], dim=-1)
        v = torch.matmul(T, rest_shape_h[:, :, :, None])[:, :, :3, 0]
        return v

    def get_vertex_transformation(self, pose, beta):
        device = pose.device
        batch_size = pose.shape[0]
        v_template = self.v_template[None, :]
        shapedirs = self.shapedirs.view(-1, 300)[None, :].expand(batch_size, -1, -1)
        beta = beta[:, :, None]
        v_shaped = torch.matmul(shapedirs, beta).view(batch_size, -1, 3) + v_template
        # batched sparse matmul not supported in pytorch
        J = []
        for i in range(batch_size):
            J.append(torch.matmul(self.J_regressor, v_shaped[i]))
        J = torch.stack(J, dim=0)
        # input it rotmat: (bs,24,3,3)
        if pose.ndimension() == 4:
            R = pose
        # input it rotmat: (bs,72)
        elif pose.ndimension() == 2:
            pose_cube = pose.view(-1, 3)  # (batch_size * 24, 1, 3)
            R = rodrigues(pose_cube).view(batch_size, 24, 3, 3)
            R = R.view(batch_size, 24, 3, 3)
        I_cube = torch.eye(3)[None, None, :].to(device)
        # I_cube = torch.eye(3)[None, None, :].expand(theta.shape[0], R.shape[1]-1, -1, -1)
        lrotmin = (R[:, 1:, :] - I_cube).view(batch_size, -1)
        posedirs = self.posedirs.view(-1, 207)[None, :].expand(batch_size, -1, -1)
        v_posed = v_shaped + torch.matmul(posedirs, lrotmin[:, :, None]).view(batch_size, -1, 3)
        J_ = J.clone()
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parent, :]
        G_ = torch.cat([R, J_[:, :, :, None]], dim=-1)
        pad_row = torch.FloatTensor([0, 0, 0, 1]).to(device).view(1, 1, 1, 4).expand(batch_size, 24,
                                                                                     -1, -1)
        G_ = torch.cat([G_, pad_row], dim=2)
        G = [G_[:, 0].clone()]
        for i in range(1, 24):
            G.append(torch.matmul(G[self.parent[i - 1]], G_[:, i, :, :]))
        G = torch.stack(G, dim=1)

        rest = torch.cat([J, torch.zeros(batch_size, 24, 1).to(device)], dim=2).view(batch_size, 24,
                                                                                     4, 1)
        zeros = torch.zeros(batch_size, 24, 4, 3).to(device)
        rest = torch.cat([zeros, rest], dim=-1)
        rest = torch.matmul(G, rest)
        G = G - rest
        T = torch.matmul(
            self.weights, G.permute(1, 0, 2, 3).contiguous().view(24, -1)).view(
            -1, batch_size, 4, 4).transpose(0, 1)
        return T

    def get_smpl_joints(self, vertices):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 38, 3)
        """
        joints = torch.einsum('bik,ji->bjk', [vertices, self.J_regressor])
        return joints

    def get_root(self, vertices):
        """
        This method is used to get the root locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 1, 3)
        """
        joints = torch.einsum('bik,ji->bjk', [vertices, self.J_regressor])
        return joints[:, 0:1, :]


if __name__ == '__main__':
    
    v_template = np.load("/home/yxiu/BigDisk/DCPIFu_data/cape/minimal_body_shape/00134/00134_minimal.npy")
    smpl = TetraSMPL('../data/SMPL_FEMALE.pkl', 
                     '../data/tetra_female_smpl.npz',
                     v_template=v_template)
    
    pose = np.load("/home/yxiu/BigDisk/DCPIFu_data/cape/npz/00134-longlong-athletics_trial1-000010.npz"
                   )["pose.npy"]
    betas = np.zeros((300,))
    pose = torch.from_numpy(pose[None,...]).float()
    betas = torch.from_numpy(betas[None,...]).float()

    vs = smpl.forward(pose, betas)
    vs = vs.detach().cpu().numpy()[0]
    ts = smpl.tetrahedrons.cpu().numpy()
    
    import trimesh
    trimesh.Trimesh(vs, ts).show()