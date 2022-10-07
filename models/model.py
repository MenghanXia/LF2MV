import torch
import torch.nn as nn
from models import basic
from .network import ArmedHourGlass2, HourGlass2, HourGlass2R, ResNet, DispOccNet, FillNet
from collections import OrderedDict
import pdb


class EncodeNet(nn.Module):
    def __init__(self, inChannel=3, outChannel=3):
        super(EncodeNet, self).__init__()
        self.net = ArmedHourGlass2(inChannel=inChannel, outChannel=outChannel)

    def forward(self, input_tensor):
        output_tensor = self.net(input_tensor)
        output_tensor = torch.tanh(output_tensor)
        return output_tensor  


class DecodeNet(nn.Module):
    def __init__(self, frameNum=49, metaNum=1):
        super(DecodeNet, self).__init__()
        self.net = ArmedHourGlass2(inChannel=metaNum+3, outChannel=3*frameNum)

    def forward(self, central_view, meta_data, frameNos):
        input_tensor = torch.cat((central_view, meta_data), dim=1)
        output_tensor = self.net(input_tensor)
        output_tensor = torch.tanh(output_tensor)
        output_warped, output_mask = None, None
        return output_tensor, output_warped, output_mask


class DecodeNetFM(nn.Module):
    def __init__(self, frameNum=49, metaNum=1, maxDisp=7.0):
        super(DecodeNetFM, self).__init__()
        self.frameNum = frameNum
        self.meta_num = metaNum
        self.max_disp = maxDisp
        self.separator = HourGlass2(inChannel=metaNum, outChannel=frameNum, resNum=3)
        self.warper = ResNet(inChannel=1, outChannel=2, resNum=5)
        self.refiner = ResNet(inChannel=4, outChannel=3, resNum=5)

    def forward(self, central_view, meta_data, frameNos):
        meta_list = self.separator(meta_data)
        label = 0
        for index in frameNos:
            meta_map = meta_list[:,index,:,:].unsqueeze(1)
            #! disparity map: should we really need to input the c_view along?
            disparity_map = self.warper(meta_map)
            grid = self.max_disp * torch.tanh(disparity_map)
            warp_map, warp_mask = basic.flowWarp(central_view, grid)
            #! refinement with disocclusion
            residual_map = self.refiner(torch.cat((warp_map, meta_map), dim=1))
            frame_map = warp_map + residual_map
            # disparity_vis = (disparity_map[:,0:1,:,:] + disparity_map[:,1:2,:,:]) / 2.0
            if label == 0:
                output_tensor = frame_map
                output_warped = warp_map
                output_mask = warp_mask
                label = 1
            else:
                output_tensor = torch.cat((output_tensor, frame_map), dim=1)
                output_warped = torch.cat((output_warped, warp_map), dim=1)
                output_mask = torch.cat((output_mask, warp_mask), dim=1)
        output_tensor = torch.tanh(output_tensor)
        return output_tensor, output_warped, output_mask


class DecodeNetFM_E(nn.Module):
    def __init__(self, frameNum=49, metaNum=1):
        super(DecodeNetFM_E, self).__init__()
        self.meta_num = metaNum
        self.separator = HourGlass2(inChannel=metaNum, outChannel=frameNum, resNum=3)
        self.warper = ResNet(inChannel=1, outChannel=2, resNum=5)
        self.refiner = ResNet(inChannel=4, outChannel=3, resNum=5)

    def forward(self, c_edit, c_view, meta_data, frameNos):
        meta_list = self.separator(meta_data)
        label = 0
        for index in frameNos:
            meta_map = meta_list[:,index,:,:].unsqueeze(1)
            #! disparity map: should we really need to input the c_view along?
            disparity_map = self.warper(meta_map)
            grid = 7.0*torch.tanh(disparity_map)
            warp_view, _ = basic.flowWarp(c_view, grid)
            warp_edit, _ = basic.flowWarp(c_edit, grid)
            #! refinement with disocclusion
            residual_map = self.refiner(torch.cat((warp_edit, meta_map), dim=1))
            frame_map = warp_edit + residual_map
            # disparity_vis = (disparity_map[:,0:1,:,:] + disparity_map[:,1:2,:,:]) / 2.0
            if label == 0:
                output_tensor = frame_map
                output_warped = warp_view
                label = 1
            else:
                output_tensor = torch.cat((output_tensor, frame_map), dim=1)
                output_warped = torch.cat((output_warped, warp_view), dim=1)
        output_tensor = torch.tanh(output_tensor)
        return output_tensor, output_warped


class DecodeNetDM_E(nn.Module):
    def __init__(self, frameNum=49, metaNum=1):
        super(DecodeNetDM_E, self).__init__()
        self.frameNum = frameNum
        self.meta_num = metaNum
        self.SepNet = HourGlass2R(inChannel=metaNum, outChannel=frameNum*2)
        self.DispNet = ResNet(inChannel=1, outChannel=2, resNum=3)
        self.FusNet = ResNet(inChannel=5, outChannel=3, resNum=5)

    def forward(self, c_edit, meta_data, frameNos):
        N, _, H, W = meta_data.shape
        output_masks = torch.ones(N, 1*len(frameNos), H, W, device=meta_data.device)
        output_disparitys = torch.ones(N, 2*(1+len(frameNos)), H, W, device=meta_data.device)
        output_frames = torch.ones(N, 3*len(frameNos), H, W, device=meta_data.device)
        #! SepNet
        meta_list = self.SepNet(meta_data)
        c_disp_feature = meta_list[:,2*(self.frameNum//2)+0,:,:].unsqueeze(1)
        c_disp = self.DispNet(c_disp_feature)
        c_disp = 5.0*torch.tanh(c_disp)
        #! defaut: the disparity of c_view in the first position
        output_disparitys[:,0:2,:,:] = c_disp
        no = 0
        for index in frameNos:
            t = (index // 7) - 3       # t is along the y axis of image
            s = (index % 7) - 3        # s is along the x axis of image
            disp_feature = meta_list[:,2*index+0,:,:].unsqueeze(1)
            fine_feature = meta_list[:,2*index+1,:,:].unsqueeze(1)
            if s == 0 and t == 0:
                #! impossible case in training: c_view will never be sampled in the 'frameNos'
                output_frames[:,3*no:3*(no+1),:,:] = c_edit
                no = no + 1
                continue
            else:
                i_disp = self.DispNet(disp_feature)
                i_disp = 5.0*torch.tanh(i_disp)
                c2i_dispmap = torch.cat((-s*i_disp[:,0:1,:,:], -t*i_disp[:,1:2,:,:]), dim=1)
                i_warped = basic.flowWarp2(c_edit, c2i_dispmap)
                with torch.no_grad():
                    occ_map = basic.checkLRDispErrorMap(i_disp, c_disp, -s, -t)
                residual_map = self.FusNet(torch.cat((i_warped, occ_map, fine_feature), dim=1))
                frame_map = i_warped + residual_map
                #! save to output tensors
                output_disparitys[:,2*(no+1):2*(no+2),:,:] = i_disp
                output_masks[:,no:no+1,:,:] = occ_map
                output_frames[:,3*no:3*(no+1),:,:] = torch.tanh(frame_map)
                no = no + 1
        return output_frames, output_disparitys, output_masks



## Ablation on decoder: baseline models -------------------------------------

class DecodeNetFM_NoSep(nn.Module):
    def __init__(self, frameNum=49, metaNum=1):
        super(DecodeNetFM_NoSep, self).__init__()
        self.frameNum = frameNum
        self.meta_num = metaNum
        self.warper = HourGlass2(inChannel=metaNum, outChannel=frameNum*2, resNum=8)
        self.refiner = ResNet(inChannel=metaNum+3*frameNum, outChannel=3*frameNum, resNum=5)

    def forward(self, central_view, meta_data, frameNos):
        disparity_maps = self.warper(meta_data)
        grid_list = 7.0*torch.tanh(disparity_maps)
        label = 0
        for index in range(0, self.frameNum):
            grid = grid_list[:,2*index:2*(index+1),:,:]
            warp_map, warp_mask = basic.flowWarp(central_view, grid)
            if label == 0:
                output_warped = warp_map
                output_mask = warp_mask
                label = 1
            else:
                output_warped = torch.cat((output_warped, warp_map), dim=1)
                output_mask = torch.cat((output_mask, warp_mask), dim=1)
        
        #! refinement with disocclusion
        residual_maps = self.refiner(torch.cat((output_warped, meta_data), dim=1))
        output_tensor = output_warped + residual_maps
        output_tensor = torch.tanh(output_tensor)
        return output_tensor, output_warped, output_mask


class DecodeNetFM_NoWarp(nn.Module):
    def __init__(self, frameNum=49, metaNum=1):
        super(DecodeNetFM_NoWarp, self).__init__()
        self.frameNum = frameNum
        self.meta_num = metaNum
        self.separator = HourGlass2(inChannel=metaNum, outChannel=frameNum, resNum=3)
        self.generator = ResNet(inChannel=4, outChannel=3, resNum=10)

    def forward(self, central_view, meta_data, frameNos):
        meta_list = self.separator(meta_data)
        label = 0
        for index in frameNos:
            meta_map = meta_list[:,index,:,:].unsqueeze(1)
            frame_map = self.generator(torch.cat((central_view, meta_map), dim=1))
            # disparity_vis = (disparity_map[:,0:1,:,:] + disparity_map[:,1:2,:,:]) / 2.0
            if label == 0:
                output_tensor = frame_map
                label = 1
            else:
                output_tensor = torch.cat((output_tensor, frame_map), dim=1)
        output_tensor = torch.tanh(output_tensor)
        output_warped, output_mask = None, None
        return output_tensor, output_warped, output_mask


