from __future__ import print_function, division
import torch, os, glob, math, random, pdb
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from utils import util


class LightFieldDataset(Dataset):

    def __init__(self, root_dir, views_num=7, crop_size=0, colorJitter=False):
        """
        Args:
            root_dir (string): directory consisting of three image folders
            transform (callable, optional): Optional transform to be applied on a sample.
        """       
        if not os.path.exists(root_dir):
            print('Warning: dataset directory %s NOT exist.'%root_dir)
            return
        lightfield_suffix = 'lightfield'
        self.input_list = util.get_dirlist(os.path.join(root_dir, lightfield_suffix))
        self.input_list.sort()
        #! get the frame num of light field
        temp = util.get_filelist(self.input_list[0])
        self.frameNum = len(temp)
        assert self.frameNum >= views_num**2
        self.views_num = views_num
        self.crop_size = crop_size
        self.root_dir = root_dir
        self.colorJitter = colorJitter

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        frame_paths = util.get_filelist(self.input_list[idx])
        inputLF_idx, targetLF_idx = [], []
        jitter_code = random.randint(1,3) if self.colorJitter else 0
        random_alpha = random.uniform(0.3,2.0)
        random_beta = random.uniform(-0.3,0.3)
        #print('jitter code:', jitter_code)
        views_num_org = int(math.sqrt(self.frameNum)+1e-5)
        radius_org = views_num_org // 2
        radius = self.views_num // 2
        if self.crop_size == 0:
            for i in range(0, self.frameNum):
                t = (i // views_num_org) - radius_org      # t is along the y axis of image
                s = (i % views_num_org) - radius_org       # s is along the x axis of image
                if abs(t) > radius or abs(s) > radius:
                    continue
                # print("t, s:", t, s)
                frame_i = Image.open(frame_paths[i]).convert("RGB")
                if jitter_code == 1:
                    result_i = transforms.functional.adjust_brightness(frame_i, random_alpha)
                elif jitter_code == 2:
                    result_i = transforms.functional.adjust_hue(frame_i, random_beta)
                elif jitter_code == 3:
                    result_i = transforms.functional.adjust_saturation(frame_i, random_alpha)
                elif jitter_code == 4:
                    result_i = transforms.functional.adjust_contrast(frame_i, random_alpha)
                else:
                    result_i = frame_i
                inputLF_idx.append(np.array(frame_i, np.float32) / 127.5 - 1.0)
                #print('length-1:', len(inputLF_idx))
                targetLF_idx.append(np.array(result_i, np.float32) / 127.5 - 1.0) 
        else:
            W, H = Image.open(frame_paths[0]).size
            y = np.random.randint(0, H - self.crop_size)
            x = np.random.randint(0, W - self.crop_size)
            for i in range(0, self.frameNum):
                t = (i // views_num_org) - radius_org      # t is along the y axis of image
                s = (i % views_num_org) - radius_org       # s is along the x axis of image
                if abs(t) > radius or abs(s) > radius:
                    continue
                # print("t, s:", t, s)
                frame_i = (Image.open(frame_paths[i]).convert("RGB")).crop((x,y,x+self.crop_size,y+self.crop_size))
                if jitter_code == 1:
                    result_i = transforms.functional.adjust_brightness(frame_i, random_alpha)
                elif jitter_code == 2:
                    result_i = transforms.functional.adjust_contrast(frame_i, random_alpha)
                elif jitter_code == 3:
                    result_i = transforms.functional.adjust_saturation(frame_i, random_alpha)
                elif jitter_code == 4:
                    result_i = transforms.functional.adjust_hue(frame_i, random_beta)
                else:
                    result_i = frame_i
                inputLF_idx.append(np.array(frame_i, np.float32) / 127.5 - 1.0)
                targetLF_idx.append(np.array(result_i, np.float32) / 127.5 - 1.0)

        #! concatenate the light-field frames into a [H,W,C] array
        #print('length:', len(inputLF_idx))
        inputLF_idx = np.concatenate(inputLF_idx, axis=2)
        inputLF_idx = torch.from_numpy(inputLF_idx.transpose((2, 0, 1)))     
        targetLF_idx = np.concatenate(targetLF_idx, axis=2)
        targetLF_idx = torch.from_numpy(targetLF_idx.transpose((2, 0, 1)))
        return {'lightfield': inputLF_idx, 'target_LF': targetLF_idx}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        sample = sample.transpose((2, 0, 1))
        #assert color_img.shape == (296,400,3), "---------------asdf" + str(color_img.shape)
        return torch.from_numpy(sample)


class RandomCrop(object):
    """Crop randomly the image in a sample."""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        h, w = sample.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        sample = sample[top: top + new_h, left: left + new_w, :]
        return sample