import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


def tensor2array(tensors):
    arrays = tensors.detach().to("cpu").numpy()
    return np.transpose(arrays, (0, 2, 3, 1))


def divideData(lightfiled_tensors, frame_num):
    view_no = 3*(frame_num//2)  #! 3*24
    c_views = lightfiled_tensors[:, view_no:view_no+3, :, :]
    index = list(range(0,view_no)) + list(range(view_no+3,frame_num*3))
    sub_frames = lightfiled_tensors[:, index, :, :]
    return c_views, sub_frames


def reshapedLF2ImgBatch(lightfield, channum=3):
    N,C,H,W = lightfield.shape
    #return torch.cat([lightfield[:, i*channum:(i+1)*channum, :, :] for i in range(C//channum)], dim=0)
    reshapredLF = torch.ones(N*(C//channum), channum, H, W, device=lightfield.device)
    for i in range(C//channum):
        reshapredLF[N*i:N*(i+1),:,:,:] = lightfield[:,channum*i:channum*(i+1),:,:]
    return reshapredLF


def XYDisparity2DispMap(lightfield_disparity):
    N,C,H,W = lightfield_disparity.shape
    disp_map = torch.ones(N, C//2, H, W, device=lightfield_disparity.device)
    #print('------', C)
    for i in range(C//2):
        disp_map[:,i,:,:] = torch.norm(lightfield_disparity[:,2*i:2*i+2,:,:], dim=1)
    return disp_map


def getSoftErrorMask(x_tensor, y_tensor):
    error_map = torch.mean(torch.abs(x_tensor-y_tensor), dim=1, keepdim=True)
    weight_map = torch.exp(-10.0*error_map)
    return 1.0-weight_map

def getSoftErrorMask(x_tensor, y_tensor, channel_unit=3):
    N,C,H,W = x_tensor.shape
    container = []
    for i in range(C//channel_unit):
        index = list(range(channel_unit*i, channel_unit*(i+1)))
        map1 = torch.abs(x_tensor[:,index,:,:]-y_tensor[:,index,:,:])
        container.append(torch.mean(map1, dim=1, keepdim=True))
    error_map = torch.cat(container, dim=1)
    weight_map = torch.exp(-10.0*error_map)
    return 1.0-weight_map

    
def flowWarp(image_batch, flow_batch):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    image_batch: [B, C, H, W] (im2)
    flow_batch: [B, 2, H, W] flow
    """
    B, C, H, W = image_batch.size()
    #! mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    baseGrid = torch.cat((xx,yy),1).float()

    if image_batch.is_cuda:
        baseGrid = baseGrid.cuda(flow_batch.device)
    vgrid = baseGrid + flow_batch

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone()/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone()/max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)
    output = nn.functional.grid_sample(image_batch, vgrid, mode='bilinear', padding_mode='zeros')
    mask = torch.ones((B,1,H,W), device=flow_batch.device)
    mask = nn.functional.grid_sample(mask, vgrid, mode='bilinear', padding_mode='zeros')
    mask = torch.round(torch.clamp(mask, 0, 1))
    
    return output, mask
   

def checkLRDepthErrorMap(depth_a, depth_b, ds, dt):
    """
    depth_a / depth_b: [B, 1, H, W]
    """
    disp_b2a = torch.cat((ds*depth_a, dt*depth_a), dim=1)
    depth_ap, _ = flowWarp(depth_b, disp_b2a)
    error_map = torch.abs(depth_a - depth_ap)
    return error_map

def checkLRDispErrorMap(disparity_a, disparity_b, ds, dt):
    """
    depth_a / depth_b: [B, 2, H, W]
    """
    disp_b2a = torch.cat((ds*disparity_a[:,0:1,:,:], dt*disparity_a[:,1:2,:,:]), dim=1)
    disparity_ap, _ = flowWarp(disparity_b, disp_b2a)
    error_map = torch.norm(disparity_ap - disparity_a, p=None, dim=1, keepdim=True)
    return error_map

    
def rgb2gray(color_batch):
    #! gray = 0.299*R+0.587*G+0.114*B
    gray_batch = color_batch[:, 0, ...] * 0.299 + color_batch[:, 1, ...] * 0.587 + color_batch[:, 2, ...] * 0.114
    gray_batch = gray_batch.unsqueeze_(1)
    return gray_batch
    
def bgr2gray(color_batch):
    #! gray = 0.299*R+0.587*G+0.114*B
    gray_batch = color_batch[:, 0, ...] * 0.114 + color_batch[:, 1, ...] * 0.587 + color_batch[:, 2, ...] * 0.299
    gray_batch = gray_batch.unsqueeze_(1)
    return gray_batch

def rgb2grayLF(color_batch):
    '''
    color_batch: [B, C, H, W]
    '''
    N,C,H,W = color_batch.shape
    container = []
    for i in range(C//3):
        r_batch = color_batch[:,3*i+0,:,:].unsqueeze(1)
        g_batch = color_batch[:,3*i+1,:,:].unsqueeze(1)
        b_batch = color_batch[:,3*i+2,:,:].unsqueeze(1)
        i_batch = 0.299*r_batch + 0.587*g_batch + 0.114*b_batch
        container.append(i_batch)
    return torch.cat(container, dim=1)


class Quantize(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x.round()
        return y

    @staticmethod
    def backward(ctx, grad_output):
        inputX = ctx.saved_tensors
        return grad_output


## -------------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.resConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        residual = self.resConv(x)
        return x + residual


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, withConvRelu=True):
        super(DownsampleBlock, self).__init__()
        if withConvRelu:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2)

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, convNum):
        super(ConvBlock, self).__init__()
        self.inConv = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        layers = []
        for _ in range(convNum - 1):
            layers.append(nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.inConv(x)
        x = self.conv(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            #nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class SkipConnection(nn.Module):
    def __init__(self, channels):
        super(SkipConnection, self).__init__()
        self.conv = nn.Conv2d(2 * channels, channels, 1, bias=False)

    def forward(self, x, y):
        x = torch.cat((x, y), 1)
        return self.conv(x)


class TransConnection(nn.Module):
    def __init__(self, channels):
        super(TransConnection, self).__init__()
        self.conv = nn.Conv2d(2 * channels, channels, 1, bias=False)        
        self.residual = ResidualBlock(channels)

    def forward(self, x, y):
        x = torch.cat((x, self.residual(y)), 1)
        return self.conv(x)


class MultiConnection(nn.Module):
    def __init__(self, channels):
        super(MultiConnection, self).__init__()
        self.conv1 = nn.Conv2d(2 * channels, channels, 1, bias=False)
        self.conv2 = nn.Conv2d(2 * channels, channels, 1, bias=False)        
        self.residual = ResidualBlock(channels)

    def forward(self, x, y, z):
        xy = torch.cat((x, y), 1)
        xy = self.residual(self.conv1(xy))
        xyz = torch.cat((xy, z), 1)
        return self.conv2(xyz)


class Space2Depth(nn.Module):
    def __init__(self, scaleFactor):
        super(Space2Depth, self).__init__()
        self.scale = scaleFactor
        self.unfold = nn.Unfold(kernel_size=scaleFactor, stride=scaleFactor)

    def forward(self, x):
        (N, C, H, W) = x.size()
        y = self.unfold(x)
        y = y.view((N, int(self.scale * self.scale), int(H / self.scale), int(W / self.scale)))
        return y
