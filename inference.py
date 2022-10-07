from __future__ import division
import os, math, datetime, time, json
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import sys
from models import basic
from models.model import *
from utils import util, datasetp, dataset, jpg_module, fidelity_metric


class Tester:
    def __init__(self, config_dict, args):
        self.batch_size = 1
        self.with_edit = args.edit 
        self.name = config_dict['name']
        self.option = config_dict['option']
        self.with_cuda = config_dict['with_cuda']
        self.meta_tolerance = config_dict['trainer']['meta_tolerance']
        print(">>> Configuring model ...")

        #! create folder to save results
        self.result_dir = os.path.abspath(args.save)
        util.exists_or_mkdir(self.result_dir)
        print("working directory: %s"%self.result_dir)
        self.frame_num = config_dict['dataset']['frame_num']
        self.meta_num = config_dict['model']['meta_num']
        print('@frame num:%d | meta num:%d | edit mode:%d'%(self.frame_num, self.meta_num, self.with_edit))
        
        #! evaluation dataset
        view_num = int(math.sqrt(self.frame_num)+1e-5)
        if self.with_edit:
            test_dataset = datasetp.LightFieldDataset(root_dir=args.input, views_num=view_num, transform=datasetp.ToTensor())
        else:
            test_dataset = dataset.LightFieldDataset(root_dir=args.input, views_num=view_num, crop_size=0, colorJitter=False)
        self.lfname_list = util.grap_dirNameList(os.path.join(args.input,'lightfield'))
        self.test_loader = DataLoader(test_dataset, self.batch_size, shuffle=False, num_workers=4)
        #! input sample name list
        if (len(self.test_loader)*self.batch_size) != len(self.lfname_list):
            print('Warning: wrong root dir to grab lightfield name list! (line:41)')

        self.encode_dir = os.path.join(self.result_dir, 'encode')
        self.restore_dir = os.path.join(self.result_dir, 'restore')
        util.exists_or_mkdir(self.encode_dir, need_remove=False)
        util.exists_or_mkdir(self.restore_dir, need_remove=False)

        #! model definition
        self.encoder = eval(config_dict['model']['encoder'])(inChannel=3*self.frame_num, outChannel=self.meta_num)
        self.max_disp = config_dict['model']['max_disp'] if 'max_disp' in config_dict['model'] else 7.0
        self.decoder = eval(config_dict['model']['decoder'])(frameNum=self.frame_num, metaNum=self.meta_num, maxDisp=self.max_disp)
        self.encoder = torch.nn.DataParallel(self.encoder).cuda()
        self.decoder = torch.nn.DataParallel(self.decoder).cuda()
        #! loading pretrained model
        assert os.path.exists(args.checkpt), "Warning: No checkpoint found!"
        self._load_pretrainedModel(args.checkpt)
        self.encoder.eval()
        self.decoder.eval()
        # quantize [-1,1] data to be {-1,1}
        self.quantizer = lambda x: basic.Quantize.apply(127.5 * (x + 1.)) / 127.5 - 1.
        self.jpeg_compressor = jpg_module.JPEG_Layer(quality=80, norm='ortho')

    
    def _test(self, best_model = False):
        print(">>> Inference with LF2MV-metaF ...")
        start_time = time.time()
        print('-------------sample num.: %d light fields.' % (len(self.test_loader)*self.batch_size))
        with torch.no_grad():
            cnt = 0
            for batch_idx, sample_batch in enumerate(self.test_loader):
                tm = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%2f')
                print('%s evaluating: [%d - %d]' % (tm, cnt, cnt+self.batch_size))
                #! depatch sample list
                cview_idx = 3*(self.frame_num//2)
                if self.with_edit:
                    input_lightfields = sample_batch['lightfield']
                    edit_cviews = sample_batch['view']
                    if self.with_cuda:
                        input_lightfields = input_lightfields.cuda()
                        edit_cviews = edit_cviews.cuda()
                else:
                    input_lightfields = sample_batch['lightfield']
                    target_lightfields = sample_batch['target_LF']
                    edit_cviews = target_lightfields[:, cview_idx:cview_idx+3, :, :]
                    if self.with_cuda:
                        input_lightfields = input_lightfields.cuda()
                        edit_cviews = edit_cviews.cuda()
                cviews = input_lightfields[:, cview_idx:cview_idx+3, :, :]

                name_list = self.lfname_list[cnt:cnt+self.batch_size]
                if self.with_edit:
                    #! encoding light field
                    pred_metas = self.encoder(input_lightfields)
                    meta_imgs = basic.tensor2array(pred_metas)
                    util.save_imagesfromBatchName(meta_imgs, self.encode_dir, name_list)
                    
                    #! restore from input
                    pred_metas = self.encoder(input_lightfields)
                    frameNos = range(0, self.frame_num)
                    restored_lightfields, coarse_maps, warp_masks = self.decoder(edit_cviews, pred_metas, frameNos)
                    #! save out images
                    restored_imgs = basic.tensor2array(restored_lightfields)
                    util.save_lightfieldsFromBatchName(restored_imgs, self.restore_dir, name_list, channel=3)
                else:
                    #! encoding halftone and restore color
                    pred_metas = self.encoder(input_lightfields)
                    save_pth = os.path.join(self.encode_dir, name_list[0].replace('.png', '.npy'))
                    meta_imgs = basic.tensor2array(pred_metas)
                    np.save(save_pth, meta_imgs)
                    frameNos = range(0, self.frame_num)
                    restored_lightfields, coarse_maps, warp_masks = self.decoder(cviews, pred_metas, frameNos) 
                    #! save out images
                    restored_imgs = basic.tensor2array(restored_lightfields)
                    util.save_lightfieldsFromBatchName(restored_imgs, self.restore_dir, name_list, channel=3)
                cnt += self.batch_size

        print("Testing finished! consumed %f sec" % (time.time() - start_time))

    
    def _evaluate_fidelity(self, restoredLF, targetLF, measuredinYUV=False):
        def RGB2YUV(rgb_img):
            yuv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)
            ## remap to [0,255] so that comput psnr and ssim
            yuv_img[:,:,1] = yuv_img[:,:,1] + 128.
            yuv_img[:,:,2] = yuv_img[:,:,2] + 128.
            return yuv_img

        N,C,H,W = restoredLF.shape
        assert N == 1
        ave_ssim, ave_psnr = 0, 0
        for i in range(self.frame_num):
            restored_img = (restoredLF[0,:,:,3*i:3*i+3] + 1.0) * 127.5
            target_img = (targetLF[0,:,:,3*i:3*i+3] + 1.0) * 127.5
            if measuredinYUV:
                restored_img = RGB2YUV(restored_img)
                target_img = RGB2YUV(target_img)
                psnr_yuv, ssim_yuv = 0, 0
                coffs = [6.0/8.0, 1.0/8.0, 1.0/8.0]
                for i in range(3):
                    #print('-----%d:'%i,np.min(restored_img[:,:,i]),np.max(restored_img[:,:,i]))
                    psnr_yuv += coffs[i] * fidelity_metric.calculate_psnr(restored_img[:,:,i], target_img[:,:,i])
                    ssim_yuv += coffs[i] * fidelity_metric.calculate_ssim(restored_img[:,:,i], target_img[:,:,i])
                psnr = psnr_yuv
                ssim = ssim_yuv
            else:
                psnr = fidelity_metric.calculate_psnr(restored_img, target_img)
                ssim = fidelity_metric.calculate_ssim(restored_img, target_img)
            ave_psnr += psnr
            ave_ssim += ssim
        ave_psnr = ave_psnr / self.frame_num
        ave_ssim = ave_ssim / self.frame_num
        return ave_psnr, ave_ssim


    def _load_pretrainedModel(self, pretrainedPath):
        if os.path.isfile(pretrainedPath) is False:
            print("@@@Warning: invalid model location & exit ...")
            return False
        device = torch.device('cuda') if self.with_cuda is True else torch.device("cpu")
        checkpoint = torch.load(pretrainedPath, map_location=device)
        self.encoder.load_state_dict(checkpoint['encoder'], strict=True)
        self.decoder.load_state_dict(checkpoint['decoder'], strict=True)
        print("[*] pretrained model loaded successfully.")            
        return True

            
if __name__ == '__main__':
    print("FLAG: %s" % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./metaFM_script.json', help='path of configuration file')
    parser.add_argument('--checkpt', type=str, default='./checkpts/model_last.pth.tar', help='path of weight')
    parser.add_argument('--edit', action='store_true', default=False, help='to load edited visual channels or not')
    parser.add_argument('--input', type=str, default='./data', help='path of data')
    parser.add_argument('--save', type=str, default='./result', help='path of result')
    args = parser.parse_args()
    if args.config is not None:
        config_dict = json.load(open(args.config))
        node = Tester(config_dict, args)
        node._test(best_model=False)
    else:
        raise Exception("Unknow --config_path")