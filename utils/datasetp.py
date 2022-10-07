from __future__ import print_function, division
import torch, os, glob, math
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from utils import util
import cv2


class LightFieldDataset(Dataset):

    def __init__(self, root_dir, views_num=7, transform=None, factor=None):
        """
        Args:
            root_dir (string): directory consisting of three image folders
            transform (callable, optional): Optional transform to be applied on a sample.
        """ 
        lightfield_suffix = 'lightfield'
        view_suffix = 'cview'
        #if not os.path.exists(os.path.join(root_dir, lightfield_suffix)):
        if not os.path.exists(root_dir):
            print('Warning: dataset directory NOT exist.')
            return       
        self.lightfield_list = util.get_dirlist(os.path.join(root_dir, lightfield_suffix))
        self.lightfield_list.sort()
        self.view_list = util.get_filelist(os.path.join(root_dir, view_suffix))
        #self.view_list = util.get_filelist('../../../Evaluation/calcByteSize/jpeg-%d_enc'%factor)
        self.view_list.sort()
        #! get the frame num of light field
        temp = util.get_filelist(self.lightfield_list[0])
        self.frameNum = len(temp)
        assert self.frameNum >= views_num**2
        self.views_num = views_num
        
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.view_list)

    def __getitem__(self, idx, max_width=1024):
        view_idx = np.array(Image.open(self.view_list[idx]).convert("RGB"), np.float32) / 127.5 - 1.0 # [H, W, 3]
        frame_paths = util.get_filelist(self.lightfield_list[idx])
        lightfield_idx = []
        views_num_org = int(math.sqrt(self.frameNum)+1e-5)
        radius_org = views_num_org // 2
        radius = self.views_num // 2
        for i in range(0, self.frameNum):
            t = (i // views_num_org) - radius_org      # t is along the y axis of image
            s = (i % views_num_org) - radius_org       # s is along the x axis of image
            if abs(t) > radius or abs(s) > radius:
                continue
            # print("t, s:", t, s)
            frame_i = np.array(Image.open(frame_paths[i]).convert("RGB"), np.float32) / 127.5 - 1.0
            lightfield_idx.append(frame_i)
        lightfield_idx = np.concatenate(lightfield_idx, axis=2)        
        sample = {'lightfield': lightfield_idx, 'view': view_idx}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):
        lightfield_frames, views = sample['lightfield'], sample['view']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        lightfield_frames = lightfield_frames.transpose((2, 0, 1))
        views = views.transpose((2, 0, 1))
        #assert color_img.shape == (296,400,3), "---------------asdf" + str(color_img.shape)
        return {'lightfield': torch.from_numpy(lightfield_frames),
                      'view': torch.from_numpy(views)}