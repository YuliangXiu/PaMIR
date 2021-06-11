import os.path as osp
import os
import torch
import numpy as np
import smplx
import sys, math
import random
from PIL import Image
import trimesh
from pdb import set_trace
from vedo import * 
from glob import glob
import torchvision.transforms as transforms

# project related libs
sys.path.append("/home/yxiu/Code/DC-PIFu")
from lib.dataset.OrthoCapeDataset import OrthoCapeDataset

class PaMIRCapeDataset(OrthoCapeDataset):
    
    def __init__(self, img_dir):
        
        self.input_size = 512
        self.root = img_dir
        self.rotations = [0]
        self.man_lst = ['03375']
        
        self.subject_list = self.get_subject_list(self.root, -1)
        
    def load_image(self, data_item):
        
        rgba = Image.open(data_item).convert('RGBA')
        self.ori_res = rgba.size[0]
        
        rgba = rgba.resize((self.input_size, self.input_size))
        mask = rgba.split()[-1]
        image = np.asarray(rgba.convert('RGB'))/255.0
        mask = np.asarray(mask)[...,None]/255.0
        image *= mask
        image += (1-mask)
        image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        
        return image

    def __getitem__(self, index):

        rid = index % len(self.rotations)
        mid = index // len(self.rotations)

        rotation = self.rotations[rid]
        subject = self.subject_list[mid]
        pid = subject.split("-")[0]
        
        if pid in self.man_lst:
            self.smpl_type = 'male'
        else:
            self.smpl_type = 'female'
        
        img_path = osp.join(self.root+"_12views", subject, "render", f"{rotation:03d}.png")
        calib_path = osp.join(self.root+"_12views", subject, "calib", f"{rotation:03d}.txt")
        npz_path = osp.join(self.root, "npz", f"{subject}.npz")
        
        data_dict = {"root_dir": self.root,
                     "subject": subject,
                     "rotation": rotation,
                     "image": self.load_image(img_path),
                     "calib": self.load_calib(calib_path)}
        
        # smpl loader
        smpl_pose = np.load(npz_path)["pose.npy"]
        smpl_trans = np.load(npz_path)["transl.npy"]
        smpl_v_template = np.load(osp.join(self.root, "minimal_body_shape", 
                                           pid, f"{pid}_minimal.npy"))
        
        data_dict.update({
            
            "trans": torch.from_numpy(smpl_trans).float(),
            "pose": torch.from_numpy(smpl_pose).float(),
            "v_template": torch.from_numpy(smpl_v_template).float(),
            "smpl_path": f"./data/SMPL_{self.smpl_type.upper()}.pkl",
            "tetra_path": f"./data/tetra_{self.smpl_type}_smpl.npz"
        })
        
        return data_dict
    
from torch.utils.data import DataLoader   
class TestingImgLoader(DataLoader):
    def __init__(self, dataset_dir, num_workers=8):
        self.dataset = PaMIRCapeDataset(dataset_dir)
        super(TestingImgLoader, self).__init__(
            self.dataset, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=True)
