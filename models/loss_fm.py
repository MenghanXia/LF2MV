import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as VM
from models import basic
from utils import filters_tensor, pytorch_ssim

eps = 0.0000001

def l2_loss(y_input, y_target):
    return F.mse_loss(y_input, y_target)


def weightedL2_loss(y_input, y_target, weight_map):
    diff_map = y_input-y_target
    diff_map = torch.mean(diff_map*diff_map, dim=1)
    return torch.sum(diff_map*weight_map) / (eps+torch.sum(weight_map))


def l1_loss(y_input, y_target):
    return F.l1_loss(y_input, y_target)


def weightedL1_loss(y_input, y_target, weight_map):
    diff_map = torch.mean(torch.abs(y_input-y_target), dim=1)
    return torch.sum(diff_map*weight_map) / (eps+torch.sum(weight_map))


def gaussianL2(yInput, yTarget):
    # data range [-1,1]
    smoother = filters_tensor.GaussianSmoothing(channels=1, kernel_size=11, sigma=2.0)
    gaussianInput = smoother(yInput)
    gaussianTarget = smoother(yTarget)
    return F.mse_loss(gaussianInput, gaussianTarget)


def ssimLoss(yInput, yTarget):
    # data range is [-1,1]
    ssim = pytorch_ssim.ssim(yInput / 2. + 0.5, yTarget / 2. + 0.5, window_size=11)
    return 1. - ssim
    

def wrapLightFieldVGGLoss(yInput, yTarget, vggLossFunc):
    # data range is [-1,1]
    N,C,H,W = yInput.shape
    batch_input = torch.cat([yInput[:, i*3:(i+1)*3, :, :] for i in range(C//3)], dim=0)
    batch_target = torch.cat([yTarget[:, i*3:(i+1)*3, :, :] for i in range(C//3)], dim=0)    
    vggLoss = vggLossFunc(batch_input/2.+0.5, batch_target/2.+0.5)
    return vggLoss


def computeDepthWarpLoss(c_depths, c_views, input_lightfields):
    N, channels, H, W = input_lightfields.shape
    frame_num = channels // 3
    warping_loss, no = 0, 0
    for i in range(frame_num):
        if i == 0:
            continue
        t = (i // 7) - 3       # t is along the y axis of image
        s = (i % 7) - 3        # s is along the x axis of image
        st_view = input_lightfields[:,3*i:3*(i+1),:,:]
        st2c_disp = torch.cat((s*c_depths, t*c_depths), dim=1)
        c_warp, _ = basic.flowWarp(st_view, st2c_disp)
        c_loss = l2_loss(c_warp, c_views)
        warping_loss += c_loss
        no += 1
    return warping_loss / no

    
def computeLRCheckLoss(disparity_b2a, disparity_a2b):
    """
    flow_batch: [B, 2, H, W] flow
    """
    disparity_b2a_est, _ = basic.flowWarp(disparity_a2b, disparity_b2a)
    disparity_a2b_est, _ = basic.flowWarp(disparity_b2a, disparity_a2b)
    deviation_a = torch.norm(disparity_b2a_est+disparity_b2a, dim=1, keepdim=True)
    deviation_b = torch.norm(disparity_a2b_est+disparity_a2b, dim=1, keepdim=True)
    #! consistency: True; occlusion: False
    mask_a = deviation_a < 0.5 + 0.01*(torch.norm(disparity_b2a_est, dim=1, keepdim=True)+\
                                                            torch.norm(disparity_b2a, dim=1, keepdim=True))
    mask_b = deviation_b < 0.5 + 0.01*(torch.norm(disparity_a2b_est, dim=1, keepdim=True)+\
                                                            torch.norm(disparity_a2b, dim=1, keepdim=True))
    checkLoss = torch.mean(deviation_a) + torch.mean(deviation_b)
    return checkLoss, mask_a, mask_b


def computeSmoothLoss(disparity_a2b, image_b):
    """
    image_batch: [B, 3, H, W], value of [-1,1]
    flow_batch: [B, 2, H, W], value of [-7,7]
    """
    guide_map = 10.0*(torch.mean(image_b, dim=1, keepdim=True)+1.0)
    gy_disp = torch.norm(disparity_a2b[:,:,:-1,:]-disparity_a2b[:,:,1:,:], dim=1)
    gx_disp = torch.norm(disparity_a2b[:,:,:,:-1]-disparity_a2b[:,:,:,1:], dim=1)
    y_weight = torch.exp(-torch.abs(guide_map[:,:,:-1,:]-guide_map[:,:,1:,:]))
    x_weight = torch.exp(-torch.abs(guide_map[:,:,:,:-1]-guide_map[:,:,:,:1]))
    smoothLoss = torch.mean(gy_disp*y_weight) + torch.mean(gx_disp*x_weight)
    return smoothLoss


def computeMaskPriorLoss(mask_batch, guidance_batch):
    """
    image_batch: [B, 1, H, W], value of [-1,1]
    guidance_batch: [B, 1, H, W], value of [0,2]  threshold_value: 0.2
    """
    weight_map = torch.exp(-10.0*guidance_batch)
    return torch.sum(mask_batch*weight_map) / (eps+torch.sum(weight_map))


def computeTotalVariation(flow_batch):
    """
    flow_batch: [B, 2, H, W] flow, value of [-7,7]
    """
    y_tv = torch.pow((flow_batch[:,:,1:,:]-flow_batch[:,:,:-1,:]),2).mean()
    x_tv = torch.pow((flow_batch[:,:,:,1:]-flow_batch[:,:,:,:-1]),2).mean()
    totalVar = x_tv + y_tv
    return totalVar


class Vgg19Loss:
    def __init__(self, multiGpu=True):
        # data in RGB format, [0,1] range
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        vgg19 = VM.vgg19(pretrained=True)
        # maxpoll after conv4_4
        self.featureExactor = nn.Sequential(*list(vgg19.features)[:28])
        for param in self.featureExactor.parameters():
            param.requires_grad = False
        if multiGpu:
            self.featureExactor = torch.nn.DataParallel(self.featureExactor).cuda()
        #! evaluation mode
        self.featureExactor.eval()
        print('[*] Vgg19Loss init!')

    def normalize(self, tensor):
        tensor = tensor.clone()
        mean = torch.as_tensor(self.mean, dtype=torch.float32, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=torch.float32, device=tensor.device)
        tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        return tensor

    def __call__(self, yInput, yTarget):
        inFeature = self.featureExactor(self.normalize(yInput))
        targetFeature = self.featureExactor(self.normalize(yTarget))
        return l2_loss(inFeature, targetFeature)
