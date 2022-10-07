from __future__ import division
from __future__ import print_function
import os, glob, shutil, math, json, pdb
import numpy as np
from PIL import Image


def save_lightfieldsFromBatchName(img_batch, save_dir, name_list, channel=3):
    if channel == 3:
        #! rgb color image
        frame_num = int(img_batch.shape[3]/3)
        for i in range(img_batch.shape[0]):
            sub_dir = os.path.join(save_dir, name_list[i])
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)
            for j in range(frame_num):
                # [-1,1] >>> [0,255]
                image = Image.fromarray((127.5*(img_batch[i, :, :, 3*j:3*j+3]+1)+0.5).astype(np.uint8))
                image.save(os.path.join(sub_dir, 'sai_%02d.png' % (j+1)), 'PNG')
    else:
        #! gray image
        frame_num = int(img_batch.shape[3])
        for i in range(img_batch.shape[0]):
            sub_dir = os.path.join(save_dir, name_list[i])
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)
            for j in range(frame_num):
                # [-1,1] >>> [0,255]
                #image = Image.fromarray((127.5*(img_batch[i, :, :, j]/9.8)+127.5).astype(np.uint8))
                image = Image.fromarray((127.5*(img_batch[i, :, :, j]+1)+0.5).astype(np.uint8))
                image.save(os.path.join(sub_dir, 'sai_%02d.png' % (j+1)), 'PNG')    
    return None


def save_imagesfromBatchName(img_batch, save_dir, name_list, jpeg_fct=None):
    if img_batch.shape[-1] == 3:
        #! rgb color image
        for i in range(img_batch.shape[0]):
            # [-1,1] >>> [0,255]
            image = Image.fromarray((127.5*(img_batch[i, :, :, :]+1)).astype(np.uint8))
            if jpeg_fct is None:
                image.save(os.path.join(save_dir, name_list[i]+'.png'), 'PNG')
            else:
                image.save(os.path.join(save_dir, name_list[i]+'.jpg'), 'JPEG', quality=jpeg_fct)
    else:
        #! single-channel gray image
        for i in range(img_batch.shape[0]):
            # [-1,1] >>> [0,255]
            image = Image.fromarray((127.5*(img_batch[i, :, :, 0]+1)).astype(np.uint8))
            if jpeg_fct is None:
                image.save(os.path.join(save_dir, name_list[i]+'.png'), 'PNG')
            else:
                image.save(os.path.join(save_dir, name_list[i]+'.jpg'), 'JPEG', quality=jpeg_fct)
    return None


def save_lightfields_from_batch(img_batch, save_dir, init_no, channel=3):
    if channel == 3:
        #! rgb color image
        frame_num = int(img_batch.shape[3]/3)
        for i in range(img_batch.shape[0]):
            sub_dir = os.path.join(save_dir, 'resultLF%03d' % (init_no + i+1))
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)
            for j in range(frame_num):
                # [-1,1] >>> [0,255]
                image = Image.fromarray((127.5*(img_batch[i, :, :, 3*j:3*j+3]+1)+0.5).astype(np.uint8))
                image.save(os.path.join(sub_dir, 'sai_%02d.png' % (j+1)), 'PNG')
    else:
        #! gray image
        frame_num = int(img_batch.shape[3])
        for i in range(img_batch.shape[0]):
            sub_dir = os.path.join(save_dir, 'resultLF%03d' % (init_no + i+1))
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)
            for j in range(frame_num):
                # [-1,1] >>> [0,255]
                #image = Image.fromarray((127.5*(img_batch[i, :, :, j]/9.8)+127.5).astype(np.uint8))
                image = Image.fromarray((127.5*(img_batch[i, :, :, j]+1)+0.5).astype(np.uint8))
                image.save(os.path.join(sub_dir, 'sai_%02d.png' % (j+1)), 'PNG')    
    return None


def save_images_from_batch(img_batch, save_dir, init_no):
    if img_batch.shape[-1] == 3:
        #! rgb color image
        for i in range(img_batch.shape[0]):
            # [-1,1] >>> [0,255]
            image = Image.fromarray((127.5*(img_batch[i, :, :, :]+1)).astype(np.uint8))
            image.save(os.path.join(save_dir, 'resultIM%03d.png' % (init_no + i+1)), 'PNG')
    else:
        #! single-channel gray image
        for i in range(img_batch.shape[0]):
            # [-1,1] >>> [0,255]
            image = Image.fromarray((127.5*(img_batch[i, :, :, 0]+1)).astype(np.uint8))
            image.save(os.path.join(save_dir, 'resultIM%03d.png' % (init_no + i+1)), 'PNG')
    return None


def save_halftones_from_batch(img_batch, save_dir, init_no):
    if img_batch.shape[-1] == 3:
        #! rgb color image
        for i in range(img_batch.shape[0]):
            # quantize [-1,1] to {0,1}
            im = img_batch[i, :, :, :]
            bitone = np.ones_like(im) * (im > 0) 
            image = Image.fromarray((255*bitone).astype(np.uint8))
            image.save(os.path.join(save_dir, 'resultH%03d.png' % (init_no + i)), 'PNG')
    else:
        #! single-channel gray image
        for i in range(img_batch.shape[0]):
            # quantize [-1,1] to {0,1}
            im = img_batch[i, :, :, 0]
            bitone = np.ones_like(im) * (im > 0) 
            image = Image.fromarray((255*bitone).astype(np.uint8))
            image.save(os.path.join(save_dir, 'resultH%03d.png' % (init_no + i)), 'PNG')
    return None
    

def save_compute_from_batch(img_batch, save_dir, cache_dir, init_no):
    num = img_batch.shape[0]
    mean_quanti = 0
    if img_batch.shape[-1] == 3:
        #! rgb color image
        for i in range(num):
            # [-1,1] >>> [0,255]
            image = Image.fromarray((127.5*(img_batch[i, :, :, :]+1)+0.5).astype(np.uint8))
            image.save(os.path.join(save_dir, 'result%05d.png' % (init_no + i+1)), 'PNG')
        return None
    else:
        #! single-channel gray image
        for i in range(num):
            # [-1,1] >>> [0,255]
            rescale = 127.5*(img_batch[i, :, :, 0]+1)
            quanti = (rescale + 0.5).astype(np.uint8)
            diff_map = np.abs(rescale-quanti)
            quanti_loss = np.mean(diff_map)
            mean_quanti += quanti_loss
            diff_img = Image.fromarray((200*diff_map).astype(np.uint8))
            image = Image.fromarray(quanti)
            diff_img.save(os.path.join(cache_dir, 'diff%05d.png' % (init_no + i+1)), 'PNG')
            image.save(os.path.join(save_dir, 'result%05d.png' % (init_no + i+1)), 'PNG')

        return mean_quanti/num


def grap_dirNameList(data_dir):
    dir_list = get_dirlist(data_dir)
    name_list = []
    for dir_path in dir_list:
        _, dir_name = os.path.split(dir_path)
        name_list.append(dir_name)
    name_list.sort()
    return name_list


def get_filelist(data_dir):
    file_list = glob.glob(os.path.join(data_dir, '*.*'))
    file_list.sort()
    return file_list

    
def get_dirlist(path):
    list = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file in files:
            m = os.path.join(path,file)
            if (os.path.isdir(m)):
                list.append(m)
    list.sort()
    return list


def save_list(save_path, data_list, append_mode=False, update_step=5):
    n = len(data_list)
    if append_mode:
        with open(save_path, 'a') as f:
            f.writelines([str(data_list[i]) + '\n' for i in range(n-update_step,n)])
    else:
        with open(save_path, 'w') as f:
            f.writelines([str(data_list[i]) + '\n' for i in range(n)])
    
    
def save_dict(save_path, dict):
    json.dumps(dict, open(save_path,"w"))
    return None


def exists_or_mkdir(path, need_remove=False):
    if not os.path.exists(path):
        os.makedirs(path)
    elif need_remove:
        shutil.rmtree(path)
        os.makedirs(path)
    return None

		
def compute_color_psnr(im_batch1, im_batch2):
    mean_psnr = 0
    im_batch1 = im_batch1.squeeze()
    im_batch2 = im_batch2.squeeze()
    num = im_batch1.shape[0]
    for i in range(num):
        # Convert pixel value to [0,255]
        im1 = 127.5 * (im_batch1[i]+1)
        im2 = 127.5 * (im_batch2[i]+1)
        #print(im1.shape)
        psnr1 = calc_psnr(im1[:,:,0], im2[:,:,0])
        psnr2 = calc_psnr(im1[:,:,1], im2[:,:,1])
        psnr3 = calc_psnr(im1[:,:,2], im2[:,:,2])
        mean_psnr += (psnr1+psnr2+psnr3) / 3.0
    return mean_psnr/num

	
def measure_quality(im_batch1, im_batch2):
    mean_psnr = 0
    mean_ssim = 0
    im_batch1 = im_batch1.squeeze()
    im_batch2 = im_batch2.squeeze()
    num = im_batch1.shape[0]
    for i in range(num):
        # Convert pixel value to [0,255]
        im1 = 127.5 * (im_batch1[i]+1)
        im2 = 127.5 * (im_batch2[i]+1)
        psnr = calc_psnr(im1, im2)
        ssim = calc_ssim(im1, im2)
        mean_psnr += psnr
        mean_ssim += ssim
    return mean_psnr/num, mean_ssim/num


def measure_psnr(im_batch1, im_batch2):
    mean_psnr = 0
    num = im_batch1.shape[0]
    for i in range(num):
        # Convert pixel value to [0,255]
        im1 = 127.5 * (im_batch1[i]+1)
        im2 = 127.5 * (im_batch2[i]+1)
        psnr = calc_psnr(im1, im2)
        mean_psnr += psnr
    return mean_psnr/num


def calc_psnr(im1, im2):
    '''
    Notice: Pixel value should be convert to [0,255]
    '''
    if im1.shape[-1] != 3:
        g_im1 = im1.astype(np.float32)
        g_im2 = im2.astype(np.float32)
    else:
        g_im1 = np.array(Image.fromarray(im1).convert('L'), np.float32)
        g_im2 = np.array(Image.fromarray(im2).convert('L'), np.float32)

    mse = np.mean((g_im1 - g_im2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def calc_ssim(im1, im2):
    """
    This function attempts to mimic precisely the functionality of ssim.m a
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    '''
    Notice: Pixel value should be convert to [0,255]
    '''
    if np.max(im1) <= 1.0:
        print('Error: pixel value should be converted to [0,255] !')
        return None
    if im1.shape[-1] != 3:
        g_im1 = im1.astype(np.float32)
        g_im2 = im2.astype(np.float32)
    else:
        g_im1 = np.array(Image.fromarray(im1).convert('L'), np.float32)
        g_im2 = np.array(Image.fromarray(im2).convert('L'), np.float32)

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255  # bitdepth of image
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = signal.fftconvolve(window, g_im1, mode='valid')
    mu2 = signal.fftconvolve(window, g_im2, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(window, g_im1 * g_im1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, g_im2 * g_im2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, g_im1 * g_im2, mode='valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()