import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as VM
from Models import basic
from Utils import util, tool

eps = 0.0000001

def l2_loss(y_input, y_target):
    return F.mse_loss(y_input, y_target)


def weightedL2_loss(y_input, y_target, weight_map):
    N,C,H,W = y_input.shape
    _,C1,_,_ = weight_map.shape
    interval = C//C1
    diff_map = y_input-y_target
    diff_map = torch.cat([torch.mean(torch.abs(diff_map[:,interval*i:interval*(i+1),:,:]), dim=1, keepdim=True) for i in range(0,C1)], dim=1)
    return torch.sum(diff_map*diff_map*weight_map) / (eps+torch.sum(weight_map))


def l1_loss(y_input, y_target):
    return F.l1_loss(y_input, y_target)


def weightedL1_loss(y_input, y_target, weight_map):
    N,C,H,W = y_input.shape
    _,C1,_,_ = weight_map.shape
    interval = C//C1
    diff_map = y_input-y_target
    diff_map = torch.cat([torch.mean(torch.abs(diff_map[:,interval*i:interval*(i+1),:,:]), dim=1, keepdim=True) for i in range(0,C1)], dim=1)
    return torch.sum(diff_map*weight_map) / (eps+torch.sum(weight_map))


def ssimLoss(yInput, yTarget):
    # data range is [-1,1]
    ssim = tool.ssim(yInput / 2. + 0.5, yTarget / 2. + 0.5, window_size=11)
    return 1. - ssim
    

def wrapLightFieldVGGLoss(yInput, yTarget, vggLossFunc):
    # data range is [-1,1]
    N,C,H,W = yInput.shape
    batch_input = torch.cat([yInput[:, i*3:(i+1)*3, :, :] for i in range(C//3)], dim=0)
    batch_target = torch.cat([yTarget[:, i*3:(i+1)*3, :, :] for i in range(C//3)], dim=0)    
    vggLoss = vggLossFunc(batch_input/2.+0.5, batch_target/2.+0.5)
    return vggLoss


def depth_warp_loss(ray_depths, input_lightfields, c_views):
    N, frame_num, H, W = ray_depths.shape
    warping_loss, no = 0, 0
    c_depth = ray_depths[:,frame_num//2,:,:].unsqueeze(1)
    #! warping loss: central view
    for t in [-2, 2]:       # s is along the x axis of image
        for s in [-2, 2]:   # t is along the y axis of image   
            index = (t+3)*7+(s+3)
            st_view = input_lightfields[:,3*index:3*(index+1),:,:]
            st2c_disp = torch.cat((s*c_depth, t*c_depth), dim=1)
            c_warp, _ = basic.flowWarp(st_view, st2c_disp)
            c_loss = l2_loss(c_warp, c_views)
            warping_loss += c_loss
            no += 1
            #! output test
            #meta_imgs = basic.tensor2array(c_warp)
            #util.save_images_from_batch(meta_imgs, './caca/', no*1)
    #! warping loss: surround subviews
    occlusion_maps = torch.ones(N, 1*frame_num, H, W, device=c_views.device)
    warped_frames = torch.ones(N, 3*frame_num, H, W, device=c_views.device)
    for i in range(frame_num):
        t = (i // 7) - 3       # t is along the y axis of image
        s = (i % 7) - 3        # s is along the x axis of image
        if s == 0 and t == 0:
            c_errormap = torch.zeros_like(c_depth, device=c_views.device)
            occlusion_maps[:,i:i+1,:,:] = c_errormap
            warped_frames[:,3*i:3*(i+1),:,:] = c_views
        else:
            i_depth = ray_depths[:,i:i+1,:,:]
            c2i_disp = torch.cat((-s*i_depth, -t*i_depth), dim=1)
            i_warp, _ = basic.flowWarp(c_views, c2i_disp)
            i_loss = l2_loss(i_warp, input_lightfields[:,3*i:3*(i+1),:,:])
            warping_loss += i_loss
            no += 1            
            i_errormap = basic.checkLRDepthErrorMap(i_depth, c_depth, -s, -t)
            occlusion_maps[:,i:i+1,:,:] = i_errormap
            warped_frames[:,3*i:3*(i+1),:,:] = i_warp
            #! output test
            #meta_imgs = basic.tensor2array(i_warp)
            #util.save_images_from_batch(meta_imgs, './caca/', no*1)
    return warping_loss / no, occlusion_maps, warped_frames


def depth_consistency_loss(ray_depths):
    _, frame_num, _, _ = ray_depths.shape
    consistency_loss, no = 0, 0
    for t in range(-3,3):      # t is along the y axis of image：-3~2
        for s in range(-3,3):  # s is along the x axis of image：-3~2
            index = (t+3)*7+(s+3)
            i_depth = ray_depths[:,index,:,:].unsqueeze(1)
            # d(s,t) of half its neighborhood views
            d10_depth = ray_depths[:,index+1,:,:].unsqueeze(1)
            d01_depth = ray_depths[:,index+7,:,:].unsqueeze(1)
            d11_depth = ray_depths[:,index+8,:,:].unsqueeze(1)
            d10_loss = basic.checkLRDepthErrorMap(i_depth, d10_depth, 1, 0)
            d01_loss = basic.checkLRDepthErrorMap(i_depth, d01_depth, 0, 1)
            d11_loss = basic.checkLRDepthErrorMap(i_depth, d11_depth, 1, 1)
            consistency_loss += (d01_loss.mean() + d10_loss.mean() + d11_loss.mean()) / 3.0
            no += 1
    return consistency_loss / no


def computeDispWarpLoss(disparity_maps, input_lightfields, c_view):
    N, dframe_num, H, W = disparity_maps.shape
    frame_num = dframe_num//2   # 49
    cview_no = frame_num//2     # 24
    c_disp = disparity_maps[:,2*cview_no:2*(cview_no+1),:,:]
    warping_loss, no = 0, 0
    #! warping loss: central view
    for t in [-2, 2]:       # s is along the x axis of image
        for s in [-2, 2]:   # t is along the y axis of image   
            index = (t+3)*7+(s+3)
            st_view = input_lightfields[:,3*index:3*(index+1),:,:]
            st2c_dispmap = torch.cat((s*c_disp[:,0:1,:,:], t*c_disp[:,1:2,:,:]), dim=1)
            c_warp, _ = basic.flowWarp(st_view, st2c_dispmap)
            c_loss = l2_loss(c_warp, c_view)
            warping_loss += c_loss
            no += 1
            #! output test
            #meta_imgs = basic.tensor2array(c_warp)
            #util.save_images_from_batch(meta_imgs, './caca/', no*1)
    #! warping loss: surround subviews
    for i in range(frame_num):
        t = (i // 7) - 3       # t is along the y axis of image
        s = (i % 7) - 3        # s is along the x axis of image
        if s == 0 and t == 0:
            # c_errormap = torch.zeros((N,1,H,W), device=c_views.device)
            continue
        else:
            i_disp = disparity_maps[:,2*i:2*(i+1),:,:]
            c2i_dispmap = torch.cat((-s*i_disp[:,0:1,:,:], -t*i_disp[:,1:2,:,:]), dim=1)
            i_warp, _ = basic.flowWarp(c_view, c2i_dispmap)
            i_loss = l2_loss(i_warp, input_lightfields[:,3*i:3*(i+1),:,:])
            warping_loss += i_loss
            no += 1
    return warping_loss / no


def computeDispWarpLossNew(disparity_maps, input_lightfields, c_view, frameNos):
    N, dframe_num, H, W = disparity_maps.shape
    frame_num = dframe_num//2   # 49
    c_disp = disparity_maps[:,0:2,:,:]
    cview_loss = 0
    #! warping loss: central view
    for t in [-2, 2]:       # s is along the x axis of image
        for s in [-2, 2]:   # t is along the y axis of image   
            index = (t+3)*7+(s+3)
            st_view = input_lightfields[:,3*index:3*(index+1),:,:]
            st2c_dispmap = torch.cat((s*c_disp[:,0:1,:,:], t*c_disp[:,1:2,:,:]), dim=1)
            c_warp, _ = basic.flowWarp(st_view, st2c_dispmap)
            c_loss = l2_loss(c_warp, c_view)
            cview_loss += c_loss
            #! output test
            #meta_imgs = basic.tensor2array(c_warp)
            #util.save_images_from_batch(meta_imgs, './caca/', no*1)
    #! warping loss: surround subviews
    warping_loss = cview_loss / 4.0
    no = 1
    for i in frameNos:
        t = (i // 7) - 3       # t is along the y axis of image
        s = (i % 7) - 3        # s is along the x axis of image
        if s == 0 and t == 0:
            # c_errormap = torch.zeros((N,1,H,W), device=c_views.device)
            continue
        else:
            i_disp = disparity_maps[:,2*no:2*(no+1),:,:]
            c2i_dispmap = torch.cat((-s*i_disp[:,0:1,:,:], -t*i_disp[:,1:2,:,:]), dim=1)
            i_warp, _ = basic.flowWarp(c_view, c2i_dispmap)
            i_loss = l2_loss(i_warp, input_lightfields[:,3*i:3*(i+1),:,:])
            warping_loss += i_loss
            no += 1
    return warping_loss / no


def computeDispConsistLoss(disparity_maps):
    _, dframe_num, _, _ = disparity_maps.shape
    frame_num = dframe_num//2   # 49
    consistency_loss, no = 0, 0
    for t in range(-3,3):      # t is along the y axis of image：-3~2
        for s in range(-3,3):  # s is along the x axis of image：-3~2
            index = (t+3)*7+(s+3)
            i_disp = disparity_maps[:,2*index:2*(index+1),:,:]
            # d(s,t) of half its neighborhood views
            no10, no01, no11 = 2*(index+1), 2*(index+7), 2*(index+8)
            d10_disp = disparity_maps[:,no10:no10+2,:,:]
            d01_disp = disparity_maps[:,no01:no01+2,:,:]
            d11_disp = disparity_maps[:,no11:no11+2,:,:]
            d10_errormap = basic.checkLRDispErrorMap(i_disp, d10_disp, 1, 0)
            d01_errormap = basic.checkLRDispErrorMap(i_disp, d01_disp, 0, 1)
            d11_errormap = basic.checkLRDispErrorMap(i_disp, d11_disp, 1, 1)
            consistency_loss += (d01_errormap.mean() + d10_errormap.mean() + d11_errormap.mean()) / 3.0
            no += 1
    return consistency_loss / no


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


def computeTotalVariation(flow_batch):
    """
    flow_batch: [B, C, H, W] flow or depth
    """
    y_tv = torch.pow((flow_batch[:,:,1:,:]-flow_batch[:,:,:-1,:]),2).mean()
    x_tv = torch.pow((flow_batch[:,:,:,1:]-flow_batch[:,:,:,:-1]),2).mean()
    totalVar = (x_tv + y_tv) / 2.0
    return totalVar


def computeMaskPriorLoss(mask_batch, guidance_batch):
    """
    image_batch: [B, 1, H, W], value of [-1,1]
    guidance_batch: [B, 1, H, W], value of [0,2]  threshold_value: 0.2
    """
    weight_map = torch.exp(-10.0*guidance_batch)
    return torch.sum(mask_batch*weight_map) / (eps+torch.sum(weight_map))


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
