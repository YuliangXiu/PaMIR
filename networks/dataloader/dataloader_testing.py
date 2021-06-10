# Software License Agreement (BSD License)
#
# Copyright (c) 2019, Zerong Zheng (zzr18@mails.tsinghua.edu.cn)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the <organization> nor the
#  names of its contributors may be used to endorse or promote products
#  derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Data loader"""

from __future__ import division, print_function

import os
import glob
import math
import numpy as np
import os.path as osp
import scipy.spatial
import scipy.io as sio
import pickle as pkl
from PIL import Image
import json
import cv2 as cv
import torch
from torch.utils.data import Dataset, DataLoader
import constant
import random
from .utils import load_data_list, generate_cam_Rt
from util import obj_io


class TestingImgDataset(Dataset):
    def __init__(self, dataset_dir, img_h, img_w, smpl_data_folder='./data', white_bg=True):
        super(TestingImgDataset, self).__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.ori_res = 512
        self.dataset_dir = dataset_dir
        data_list = glob.glob(os.path.join(dataset_dir, '*.png'))
        data_list = [d for d in data_list if not d.endswith('_mask.png')]
        
        random.seed(1993)
        # self.data_list = random.sample(sorted(data_list), k=2)
        self.data_list = sorted(data_list)
        
        # self.data_list = ["/home/yxiu/BigDisk/DCPIFu_data/cape/03375/blazerlong/images/stretch_trial1_101.png"]
        
        
        self.len = len(self.data_list)
        self.white_bg = white_bg

        # load smpl model data for usage
        jmdata = np.load(os.path.join(smpl_data_folder, 'joint_model.npz'))
        self.J_dirs = jmdata['J_dirs']
        self.J_template = jmdata['J_template']

        # some default parameters for testing
        self.default_testing_cam_R = constant.cam_R
        self.default_testing_cam_t = constant.cam_t
        self.default_testing_cam_f = constant.cam_f

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        data_item = self.data_list[item]
        data_fd, img_fname = os.path.split(data_item)

        img = self.load_image(data_item)
        
        # Image.fromarray((img*255.0).astype(np.uint8)).save("test.png")

        return_dict = {
            'img_id': item,
            'img_dir': data_item,
            'img': torch.from_numpy(img.transpose((2, 0, 1))),
        }

        kpt_fname = osp.join(self.dataset_dir, "../2djoints", osp.basename(data_item)[:-4]+".json")
        if os.path.exists(kpt_fname):
            keypoints = self.load_keypoints(kpt_fname)
            return_dict.update({'keypoints': torch.from_numpy(keypoints)})

        param_fname = data_item[:-4] + '_smplparams.pkl'
        if os.path.exists(param_fname):
            pose, betas, trans, scale = self.load_smpl_parameters(param_fname)
            return_dict.update({'betas': torch.from_numpy(betas),
                                'pose': torch.from_numpy(pose),
                                'scale': torch.from_numpy(scale),
                                'trans': torch.from_numpy(trans),
                                })
        else:
            param_fname = os.path.join(data_fd, img_fname[:-4] + '_smplparams.pkl')
            if os.path.exists(param_fname):
                pose, betas, trans, scale = self.load_smpl_parameters(param_fname)
                return_dict.update({'betas': torch.from_numpy(betas),
                                    'pose': torch.from_numpy(pose),
                                    'scale': torch.from_numpy(scale),
                                    'trans': torch.from_numpy(trans),
                                    })

        mesh_fname = os.path.join(data_fd, img_fname[:-4] + '.obj')
        if os.path.exists(mesh_fname):
            mesh_v, mesh_f = self.load_mesh_vertices(mesh_fname)
            return_dict.update({
                'mesh_vert': torch.from_numpy(mesh_v),
                'mesh_face': torch.from_numpy(mesh_f)
            })

        return return_dict
    
    def load_image(self, data_item):
        
        rgba = Image.open(data_item).convert('RGBA')
        self.ori_res = rgba.size[0]
        
        rgba = rgba.resize((self.img_w, self.img_h))
        mask = rgba.split()[-1]
        image = np.asarray(rgba.convert('RGB'))/255.0
        mask = np.asarray(mask)[...,None]/255.0
        image *= mask
        image += (1-mask)
        
        return image.astype(np.float32)

    def load_smpl_parameters(self, data_item):
        with open(data_item, 'rb') as fp:
            data = pkl.load(fp, encoding='iso-8859-1')
            pose = np.float32(data['body_pose']).reshape((-1, ))
            betas = np.float32(data['betas']).reshape((-1,))
            trans = np.float32(data['global_body_translation']).reshape((1, -1))
            scale = np.float32(data['body_scale']).reshape((1, -1))
        return pose, betas, trans, scale

    def load_mesh_vertices(self, data_item):
        mesh = obj_io.load_obj_data(data_item)
        return mesh['v'].astype(np.float32), mesh['f'].astype(np.int32)
    
    def load_keypoints(self, data_item):
        with open(data_item) as fp:
            data = json.load(fp)
        keypoints = []
        
        kp_data = np.array(data['keypoints'], dtype=np.float32)
        kp_data = kp_data.reshape([-1, 3])
        kp_data = kp_data[constant.body25_to_joint]      # rearrange keypoints
        kp_data[constant.body25_to_joint < 0] *= 0.0     # remove undefined keypoints
        # kp_data[:, 0] = kp_data[:, 0]*2 / self.img_w - 1.0
        # kp_data[:, 1] = kp_data[:, 1]*2 / self.img_h - 1.0
        kp_data[:, 0] = kp_data[:, 0]*2 / self.ori_res - 1.0
        kp_data[:, 1] = kp_data[:, 1]*2 / self.ori_res - 1.0
        
        keypoints.append(kp_data)
        
        if len(keypoints) == 0:
            keypoints.append(np.zeros([24, 3]))

        return np.array(keypoints[0], dtype=np.float32)


class TestingImgLoader(DataLoader):
    def __init__(self, dataset_dir, img_h, img_w, white_bg=True, num_workers=8):
        self.dataset = TestingImgDataset(dataset_dir=dataset_dir, img_h=img_h, img_w=img_w, white_bg=white_bg)
        self.img_h = img_h
        self.img_w = img_w
        self.dataset_dir = dataset_dir
        super(TestingImgLoader, self).__init__(
            self.dataset, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=True)


