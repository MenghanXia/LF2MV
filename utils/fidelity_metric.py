from __future__ import division
from __future__ import print_function
import os, glob, shutil, math, json
import numpy as np
from PIL import Image
import cv2
import torch, lpips

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def get_filelist(data_dir):
    file_list = glob.glob(os.path.join(data_dir, '*.*'))
    file_list.sort()
    return file_list


def save_list(save_path, data_list):
    n = len(data_list)
    with open(save_path, 'w') as f:
        f.writelines([str(data_list[i]) + '\n' for i in range(n)])
    return None
    

def batch_evaluation(data_dir, gt_dir, save_name='wow', lpips_fn=None, gray_mode=False):
    file_list1 = get_filelist(data_dir)
    file_list2 = get_filelist(gt_dir)
    if (len(file_list1) != len(file_list2)):
        print('Warning: image numbers {%d & %d} NOT match!' % (len(file_list1), len(file_list2)))
        return
    num = len(file_list1)
    print('@%d images loaded.' % num)
    assert num != 0
    psnr_list, ssim_list, lpips_list = [], [], []
    psnr_ave, ssim_ave, lpips_ave = 0, 0, 0
    cnt = 0
    for idx in range(num):
        print('-img', idx+1)
        if gray_mode:
            img1 = np.array(Image.open(file_list1[idx]).convert("L"))
            img2 = np.array(Image.open(file_list2[idx]).convert("L"))
        else:
            img1 = np.array(Image.open(file_list1[idx]).convert("RGB"))
            img2 = np.array(Image.open(file_list2[idx]).convert("RGB"))
        psnr = calculate_psnr(img1, img2)
        psnr_list.append(psnr)
        psnr_ave += psnr
        ssim = calculate_ssim(img1, img2)
        ssim_list.append(ssim)
        ssim_ave += ssim
        lpips = 0.0
        if lpips_fn is not None:
            img1 = torch.from_numpy((img1 / 127.5 - 1.).transpose((2, 0, 1)))
            img2 = torch.from_numpy((img2 / 127.5 - 1.).transpose((2, 0, 1)))
            lpips = lpips_fn(img1.unsqueeze(0).float(), img2.unsqueeze(0).float())
        lpips_list.append(lpips)
        lpips_ave += lpips
        cnt += 1
    #! write out the result list
    with open('Saved/ldr_metric-%s.txt'%save_name, 'w') as f:
        f.writelines([str(psnr_list[i])+', '+str(ssim_list[i])+', '+str(lpips_list[i]) + '\n' for i in range(cnt)])
    print('Average PSNR/SSIM:%3.3f / %3.3f / %3.3f' % (psnr_ave/cnt, ssim_ave/cnt, lpips_ave/cnt))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('--dir1', type=str, default='../MultipInvTM/DEmD_LGP/Test/encode_0')
    #parser.add_argument('--dir2', type=str, default='../../0DataZoo/Dataset_HDR/ZM5300/Test/ldr_durand02')
    #! target tmo dir: ldr_reinhard02k040   ldr_durand02    ldr_li05    ldr_a14
    tmo_names = ['ldr_reinhard02k040', 'ldr_durand02', 'ldr_li05', 'ldr_a14']
    parser.add_argument('-wd', '--work_dir', type=str, default='DEmD_LGP')
    parser.add_argument('-t', '--tmo', type=int, default='1')
    parser.add_argument('-n', '--save_name', type=str, default='wow')
    parser.add_argument('--lpips', action='store_true')
    parser.add_argument('--gray', action='store_true')
    args = parser.parse_args()
    data_dir = os.path.join('../../0DataZoo/Dataset_HDR/ZM5300/Test', tmo_names[args.tmo])
    data_dir2 = os.path.join('../', args.work_dir, 'Test/encode_%d'%args.tmo)
    print('-@dir1:%s\n-@dir2:%s' % (data_dir, data_dir2))
    loss_fn_vgg = lpips.LPIPS(net='vgg') if args.lpips else None
    batch_evaluation(data_dir, data_dir2, args.save_name, lpips_fn=loss_fn_vgg, gray_mode=args.gray)
    print('Hello, world!')