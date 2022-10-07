import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import pdb

eps = 1e-7

def RGB2HSV(rgb_img):
    #! img: [N C H W]  with element value [0, 1]
    #test()
    h_img = rgb_img[:,0,:,:].clone()
    v_img = rgb_img[:,0,:,:].clone()
    s_img = rgb_img[:,0,:,:].clone()
    max_img = rgb_img.max(1)[0]
    min_img = rgb_img.min(1)[0]
    h_img[rgb_img[:,0,:,:]==max_img] = (0.0 + ((rgb_img[:,1,:,:]-rgb_img[:,2,:,:]) / (max_img-min_img+eps))[rgb_img[:,0,:,:]==max_img]) % 6
    h_img[rgb_img[:,1,:,:]==max_img] = 2.0 + ((rgb_img[:,2,:,:]-rgb_img[:,0,:,:]) / (max_img-min_img+eps))[rgb_img[:,1,:,:]==max_img]
    h_img[rgb_img[:,2,:,:]==max_img] = 4.0 + ((rgb_img[:,0,:,:]-rgb_img[:,1,:,:]) / (max_img-min_img+eps))[rgb_img[:,2,:,:]==max_img]   
    #pdb.set_trace()
    h_img[max_img==min_img] = 0.0
    h_img = h_img/6.0
    v_img = max_img
    s_img = (max_img-min_img) / (max_img+eps)    
    #print(torch.mean(h_img).data)
    return torch.stack([h_img, s_img, v_img], 1)


def HSV2RGB(hsv_img):
    #! img: [N C H W]
    r_img = hsv_img[:,0,:,:].clone()
    g_img =hsv_img[:,0,:,:].clone()
    b_img =hsv_img[:,0,:,:].clone()
    h_rd = torch.abs(torch.floor(hsv_img[:,0,:,:]*6))  #! here using abs is to avoid "-0.0" which is not equal to 0.0
    #pdb.set_trace()
    f_img = hsv_img[:,0,:,:]*6.0 - h_rd
    v_img = hsv_img[:,2,:,:]
    p_img = v_img*(1.0-hsv_img[:,1,:,:])
    q_img = v_img*(1.0-hsv_img[:,1,:,:]*f_img)
    t_img = v_img*(1.0-hsv_img[:,1,:,:]*(1.0-f_img))
    #! h_rd == 0 or 6
    r_img[h_rd==0] = v_img[h_rd==0]
    r_img[h_rd==6] = v_img[h_rd==6]
    g_img[h_rd==0] = t_img[h_rd==0]
    g_img[h_rd==6] = t_img[h_rd==6]
    b_img[h_rd==0] = p_img[h_rd==0]
    b_img[h_rd==6] = p_img[h_rd==6]
    #! h_rd == 1
    r_img[h_rd==1] = q_img[h_rd==1]
    g_img[h_rd==1] = v_img[h_rd==1]
    b_img[h_rd==1] = p_img[h_rd==1]
    #! h_rd == 2
    r_img[h_rd==2] = p_img[h_rd==2]
    g_img[h_rd==2] = v_img[h_rd==2]
    b_img[h_rd==2] = t_img[h_rd==2]
    #! h_rd == 3
    r_img[h_rd==3] = p_img[h_rd==3]
    g_img[h_rd==3] = q_img[h_rd==3]
    b_img[h_rd==3] = v_img[h_rd==3]
    #! h_rd == 4
    r_img[h_rd==4] = t_img[h_rd==4]
    g_img[h_rd==4] = p_img[h_rd==4]
    b_img[h_rd==4] = v_img[h_rd==4]
    #! h_rd == 5
    r_img[h_rd==5] = v_img[h_rd==5]
    g_img[h_rd==5] = p_img[h_rd==5]
    b_img[h_rd==5] = q_img[h_rd==5]
    return torch.stack([r_img, g_img, b_img], 1)


def adjustBrightness(hsv_img, gain, gamma):
    #! img: [N C H W]  with element value [0, 1]
    hsv_img[:, 0, :, :] = gain * torch.pow(hsv_img[:, 0, :, :], gamma)
    return hsv_img

   
def adjustHue(hsv_img, delta):
    #! img: [N C H W]
    hsv_img[:, 1, :, :] = hsv_img[:, 1, :, :] - delta
    return hsv_img


def adjustSaturation(hsv_img, factor):
    #! img: [N C H W]
    hsv_img[:, 2, :, :] = factor * hsv_img[:, 2, :, :]
    #hsv_img[:, 2, :, :] = torch.pow(hsv_img[:, 2, :, :], factor)
    return hsv_img

   
def adjustContrast(hsv_img, factor):
    #! img: [N C H W]  with element value [0, 1]
    lumi_mu = torch.mean(hsv_img[:, 0, :, :], dim=(1,2))
    lumi_mu = lumi_mu.expand(hsv_img.size()[2], hsv_img.size()[3], hsv_img.size()[0])
    #pdb.set_trace()
    lumi_mu = (lumi_mu.transpose(2, 0)).transpose(2, 1)
    #print('-----------lumi size', lumi_mu.size())
    hsv_img[:, 0, :, :] = factor * (hsv_img[:, 0, :, :] - lumi_mu) + lumi_mu
    return hsv_img
    
   
def enhanceDetail(rgb_img, factors):
    #! img: [N C H W]  with element value [0, 1]
    #! blur kernel
    channel = 3
    win_size = 5
    sigma = 1.5
    gauss = torch.as_tensor([exp(-(x - win_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(win_size)], device=rgb_img.device).unsqueeze(1)
    gauss = gauss/gauss.sum()
    win_2d = gauss.mm(gauss.t()).float().unsqueeze(0).unsqueeze(0)
    #pdb.set_trace()
    #! enhance details
    smooth_img = torch.nn.functional.conv2d(rgb_img, win_2d, padding=win_size // 2, groups=3)
    rgb_img = factors*(rgb_img-smooth_img) + smooth_img
    return rgb_img


def colorJitter(rgb_img, factors):
    #! img: [N C H W]  with element value [0, 1]
    rgb_img[:, 0, :, :] = torch.pow(rgb_img[:, 0, :, :], factors[0])
    rgb_img[:, 1, :, :] = torch.pow(rgb_img[:, 1, :, :], factors[1])
    rgb_img[:, 2, :, :] = torch.pow(rgb_img[:, 2, :, :], factors[2])
    return rgb_img
    
    
def randomInteraction(images, light_fields):
    #! RGB: [N C H W]  with element value [0, 1]
    batch_size, frame_num, _, _ = light_fields.size()
    frame_num = int(frame_num/3)
    images = images*0.5+0.5
    light_fields = light_fields*0.5+0.5
    reshaped_lightfields = torch.cat([light_fields[:, i*3:(i+1)*3, :, :] for i in range(frame_num)], dim=0)
    
    code = np.random.uniform(0, 1, 1)
    if  0.7 > 0.5:
        #! random factors of: brightness, saturation, contrast, and enhancement
        factors = torch.as_tensor(np.random.uniform(0.8, 1.2, 4).astype(np.float32), device=images.device)
        delta = torch.as_tensor(np.random.uniform(-0.10, 0.10, 1).astype(np.float32), device=images.device)
        gamma = torch.as_tensor(np.random.uniform(0.75, 2.0, 1).astype(np.float32), device=images.device)
        #print('----- factors:', factors)
        #print('----- delta:', delta)
        #print('----- gamma:', gamma)
        #! -------- manipulation in HSV space
        #! convert RGB to HSV
        hsv_lightfields = RGB2HSV(reshaped_lightfields)
        hsv_images = RGB2HSV(images)
        #! brightness
        hsv_images = adjustBrightness(hsv_images, factors[0], gamma)
        hsv_lightfields = adjustBrightness(hsv_lightfields, factors[0], gamma)
        #! satruation
        hsv_images = adjustSaturation(hsv_images, factors[1])
        hsv_lightfields = adjustSaturation(hsv_lightfields, factors[1])
        #! hue
        #hsv_images = adjustHue(hsv_images, delta)
        #hsv_lightfields = adjustHue(hsv_lightfields, delta)
        #! contrast
        #hsv_images = adjustContrast(hsv_images, factors[2])
        #hsv_lightfields = adjustContrast(hsv_lightfields, factors[2])
        #! -------- manipulation in RGB space
        #! convert HSV to RGB
        reshaped_lightfields = HSV2RGB(hsv_lightfields)
        images = HSV2RGB(hsv_images)
    else:
        #! detail enhancement
        #images = enhanceDetail(images, factors[3])
        #light_fields = enhanceDetail(light_fields, factors[3])
        #! RGB jittering
        rgb_gammas = torch.as_tensor(np.random.uniform(0.7, 1.8, 3).astype(np.float32), device=images.device)
        #print('----- rgb factors:', rgb_gammas)
        images = colorJitter(images, rgb_gammas)
        reshaped_lightfields = colorJitter(reshaped_lightfields, rgb_gammas)
        
    light_fields = torch.cat([reshaped_lightfields[i*batch_size:(i+1)*batch_size, :, :, :] for i in range(frame_num)], dim=1)
    light_fields = light_fields*2-1
    images = images*2-1
    return images, light_fields