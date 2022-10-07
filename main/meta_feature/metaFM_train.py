from __future__ import division
import os, math, datetime, time, json
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
sys.path.append("../..")
from models import basic, loss_fm
from models.model import *
from utils import util, dataset, jpg_module, img_operator
from collections import OrderedDict

class Trainer:

    def __init__(self, config_dict, resume_mode = False):
        '''Note: training takes 'with_cuda' as default'''
        #! parsing training hyper-parameters
        self.name = config_dict['name']
        self.option = config_dict['option']
        self.with_cuda = config_dict['with_cuda']
        self.seed = config_dict['trainer']['seed']
        self.hypeCoeffs = config_dict['loss_weight']
        self.n_epochs = config_dict['trainer']['n_epochs']
        self.batch_size = config_dict['trainer']['batch_size']
        self.learning_rate = config_dict['trainer']['lr']
        self.display_iters = config_dict['trainer']['display_iters']
        self.save_epochs = config_dict['trainer']['save_epochs']
        self.need_valid = config_dict['trainer']['need_valid']
        self.meta_tolerance = config_dict['trainer']['meta_tolerance']
        self.monitorMetric = 9999
        self.start_epoch = 0
        self.resume_mode = resume_mode
        #! create folders to save trained model and results
        self.work_dir = os.path.join(config_dict['save_dir'], self.name+self.option)
        print('**************** %s ****************' % (self.name+self.option))
        util.exists_or_mkdir(self.work_dir)
        self.cache_dir = os.path.join(self.work_dir, 'cache')
        util.exists_or_mkdir(self.cache_dir, need_remove=False)
        #! save config-json file to work directory
        json.dump(config_dict, open(os.path.join(self.work_dir, 'config_script.json'), "w"), indent=4, sort_keys=False)
        if self.need_valid:
            self.encode_validir = os.path.join(self.cache_dir, 'encode')
            util.exists_or_mkdir(self.encode_validir, need_remove=False)
            self.restore_validir = os.path.join(self.cache_dir, 'restore')
            util.exists_or_mkdir(self.restore_validir, need_remove=False)
        #! epoch loss list
        self.learningrateList, self.metricList, self.totalLossList = [], [], []
        self.warpL2LossList, self.restoreL2LossList = [], []
        self.frame_num = config_dict['dataset']['frame_num']
        self.meta_num = config_dict['model']['meta_num']
        print('@frame num:%d | meta num:%d'%(self.frame_num, self.meta_num))

        '''dataset preparation'''
        #! training dataset
        #transforms_train = transforms.Compose([dataset.RandomCrop(128), transforms.ToTensor()])
        view_num = int(math.sqrt(self.frame_num)+1e-5)
        training_dataset = dataset.LightFieldDataset(root_dir=config_dict['dataset']['train'], views_num=view_num, crop_size=128, colorJitter=False)
        self.train_loader = DataLoader(training_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)        
        print('-------------sample num.: %d light fields.' % len(training_dataset))
        print('@batch size:%d'%self.batch_size)
        #! valiation dataset
        valid_dataset = dataset.LightFieldDataset(root_dir=config_dict['dataset']['val'], views_num=view_num, crop_size=0)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        '''set model and optimization'''
        #! model definition
        self.encoder = eval(config_dict['model']['encoder'])(inChannel=3*self.frame_num, outChannel=self.meta_num)
        if config_dict['model']['decoder'] == 'DecodeNetFM':
            self.max_disp = config_dict['model']['max_disp'] if 'max_disp' in config_dict['model'] else 7.0
            self.decoder = eval(config_dict['model']['decoder'])(frameNum=self.frame_num, metaNum=self.meta_num, maxDisp=self.max_disp)
        else:
            self.decoder = eval(config_dict['model']['decoder'])(frameNum=self.frame_num, metaNum=self.meta_num)
        #! set target device
        if self.with_cuda:
            self.encoder = torch.nn.DataParallel(self.encoder).cuda()
            self.decoder = torch.nn.DataParallel(self.decoder).cuda()

        # optimizer
        self.encoderOptimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        self.decoderOptimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.learning_rate)

        if self.resume_mode:
            self._resume_checkpoint()   #! update "self.start_epoch" as used in the lr_scheduler
        decay_ratio = 1.0/300   # 1.0/200
        decay_epochs = self.n_epochs
        polynomial_decay = lambda epoch: 1 + (decay_ratio - 1) * ((epoch+self.start_epoch)/decay_epochs)\
            if (epoch+self.start_epoch) < decay_epochs else decay_ratio
        self.lr_encoderSheduler = torch.optim.lr_scheduler.LambdaLR(self.encoderOptimizer, lr_lambda=polynomial_decay)
        self.lr_decoderSheduler = torch.optim.lr_scheduler.LambdaLR(self.decoderOptimizer, lr_lambda=polynomial_decay)
        #self.lr_sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **config_dict['trainer']['lr_sheduler'])
        #self.lr_sheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=12, gamma=0.5)
        #self.lr_sheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30,80], gamma=0.1)
        
        '''construct key loss computation'''
        #self.vggLossFunc = loss_fm.Vgg19Loss()
        # quantize [-1,1] data to be {-1,1}
        self.quantizer = lambda x: basic.Quantize.apply(127.5 * (x + 1.)) / 127.5 - 1.
        self.jpeg_compressor = jpg_module.JPEG_Layer(quality=80, norm='ortho')
 
    def _train(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        start_time = time.time()
        for epoch in range(self.start_epoch, self.n_epochs):
            start_time_epoch = time.time()
            #! step1: iterative optimization within this epoch
            epoch_lr = self.lr_encoderSheduler.get_lr()[0]
            epochLoss = self._train_epoch(epoch) 
            #! step2: update learning rate by lr_sheduler
            self.lr_encoderSheduler.step()
            self.lr_decoderSheduler.step()
            #self.lr_sheduler.step(epochLoss)
            #epoch_lr = self.optimizer.state_dict()['param_groups'][0]['lr']            
            self.learningrateList.append(epoch_lr)
            epochMetric = self._valid_epoch(epoch) if self.need_valid else 0.0
            print("[*] ----- epoch: %d/%d | loss: %4.4f | metric: %4.4f | Time-consumed: %4.2f -----" %\
                (epoch+1, self.n_epochs, epochLoss, epochMetric, (time.time() - start_time_epoch)))
            #! step3: save the model of this epoch and the best model so far
            if ((epoch+1) % self.save_epochs == 0 or epoch == (self.n_epochs-1)):
                print('------------- saving model & loss ...')
                self._save_checkpoint(epoch)
                self._save_losses(resume_mode=self.resume_mode)
            if (epoch > (self.n_epochs // 2) and self.need_valid and self.monitorMetric > epochMetric):
                print('------------- saving best model ...')
                self.monitorMetric = epochMetric
                self._save_checkpoint(epoch, save_best=True)            
        #! displaying the training time
        print("Training finished! consumed %f sec" % (time.time() - start_time))
   
        
    def _train_epoch(self, epoch):
        #! set model to training mode
        self.encoder.train()
        self.decoder.train() 
        restoreVGGLoss, restoreL2Loss = 0, 0
        warpL2Loss = 0
        totalLoss, n_iters = 0, 0
        #for sample_batch, sample_batch1 in zip(self.train_loader, self.train_loader1):
        for batch_idx, sample_batch in enumerate(self.train_loader):
            #! reset gradient buffer to zero
            self.encoderOptimizer.zero_grad()
            self.decoderOptimizer.zero_grad()
            #! depatch sample batch
            input_lightfields = sample_batch['lightfield']
            input_lightfields = input_lightfields.cuda()

            #! forward process
            pred_metaMaps = self.encoder(input_lightfields)
            if self.meta_tolerance == 'qt':
                pred_metaMaps = self.quantizer(pred_metaMaps)
            elif self.meta_tolerance == 'jpeg':
                pred_metaMaps = self.jpeg_compressor(pred_metaMaps)
            view_no = 3*(self.frame_num//2)
            central_views = input_lightfields[:, view_no:view_no+3, :, :]
            #random_code = np.random.uniform(0, 1, 1)
            #if random_code > 0.5:
                #! interactive editing simulation
                #central_views, input_lightfields = img_operator.randomInteraction(central_views, input_lightfields)
            #! random reconstruction
            if 'noSep' in self.option:
                frameNos = range(0, self.frame_num)
            else:
                frameNos = random.sample(range(0, self.frame_num), self.frame_num // 2)
            restored_lightfields, coarse_maps, warp_masks = self.decoder(central_views, pred_metaMaps, frameNos)
            targetIndexs = np.reshape(np.array([[3*v, 3*v+1, 3*v+2] for v in frameNos]), 3*len(frameNos)).tolist()
            #! restoration loss
            #print('---------', restored_lightfields.shape)
            restoreL2Loss_idx = loss_fm.l2_loss(restored_lightfields, input_lightfields[:,targetIndexs,:,:])
            if "noWarp" in self.option:
                warpl2Loss_idx = torch.zeros_like(restoreL2Loss_idx)
            else:                
                coarseError = torch.abs(coarse_maps-input_lightfields[:,targetIndexs,:,:])
                warpl2Loss_idx = torch.sqrt(torch.sum(torch.mean(coarseError*coarseError, dim=1, keepdim=True) * \
                                                                warp_masks)/torch.sum(warp_masks))
            #print('-------', warpl2Loss_idx.item(), restoreL2Loss_idx.item())
            totalLoss_idx = self.hypeCoeffs['restoreL2Weight'] * restoreL2Loss_idx + \
                                      self.hypeCoeffs['warpL2Weight'] * warpl2Loss_idx

            #! backward process
            totalLoss_idx.backward()
            self.encoderOptimizer.step()
            self.decoderOptimizer.step()
            
            #! add to epoch losses
            restoreL2Loss += restoreL2Loss_idx.item()
            warpL2Loss += warpl2Loss_idx.item()
            totalLoss += totalLoss_idx.item()
            n_iters += 1
            if (n_iters+1) % self.display_iters == 0:
                tm = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                print("%s >> [%d/%d] iter: %d  loss: %4.4f" % (tm, epoch+1, self.n_epochs, n_iters, totalLoss_idx.item()))
                
        #! epoch data: average losses
        self.totalLossList.append(totalLoss / n_iters)
        self.restoreL2LossList.append(restoreL2Loss / n_iters)
        self.warpL2LossList.append(warpL2Loss / n_iters)
        return totalLoss / n_iters

        
    def _valid_epoch(self, epoch):
        #! set model to evaluation mode
        self.encoder.eval()
        self.decoder.eval()
        epochMetric, cnt = 0, 0
        with torch.no_grad():
            for batch_idx, sample_batch in enumerate(self.valid_loader):
                #! depatch sample list
                input_lightfields = sample_batch['lightfield']
                input_lightfields = input_lightfields.cuda()                    
                #! forward process
                pred_metaMaps = self.encoder(input_lightfields)
                view_no = 3*(self.frame_num//2)
                central_views = input_lightfields[:, view_no:view_no+3, :, :] 
                restored_lightfields, coarse_maps, warp_masks = self.decoder(central_views, pred_metaMaps, range(0, self.frame_num))
                
                #! save images
                meta_imgs = basic.tensor2array(pred_metaMaps)
                util.save_images_from_batch(meta_imgs, self.encode_validir, cnt*self.batch_size)
                restored_imgs = basic.tensor2array(restored_lightfields)
                util.save_lightfields_from_batch(restored_imgs, self.restore_validir, cnt*self.batch_size)
                #! computing metric
                metric = loss_fm.l2_loss(restored_lightfields, input_lightfields)
                epochMetric += metric.item()
                cnt += 1
            #! average metric
            epochMetric = epochMetric / cnt
            self.metricList.append(epochMetric)
        return epochMetric


    def _save_losses(self, resume_mode = False):
        util.save_list(os.path.join(self.cache_dir, "learningrate_list"), self.learningrateList, resume_mode)
        util.save_list(os.path.join(self.cache_dir, "loss_total"), self.totalLossList, resume_mode)
        util.save_list(os.path.join(self.cache_dir, "loss_restore"), self.restoreL2LossList, resume_mode)
        util.save_list(os.path.join(self.cache_dir, "loss_warp"), self.warpL2LossList, resume_mode)
        if self.need_valid:
            util.save_list(os.path.join(self.cache_dir, "metric_valid"), self.metricList, resume_mode)
            

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'monitor_best': self.monitorMetric, 
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'Eoptimizer': self.encoderOptimizer.state_dict(),
            'Doptimizer': self.decoderOptimizer.state_dict()
        }       
        save_path = os.path.join(self.work_dir, 'checkpoint-epoch{:03d}.pth.tar'.format(epoch))
        if save_best:
            save_path = os.path.join(self.work_dir, 'model_best.pth.tar')
        else:
            save_path = os.path.join(self.work_dir, 'model_last.pth.tar')
        #! save checkpoint
        torch.save(state, save_path)


    def _resume_checkpoint(self):
        resume_path = os.path.join(self.work_dir, 'model_last.pth.tar')
        if os.path.isfile(resume_path) is False:
            print("@@@Warning: invalid checkpoint location & traning from scratch ...")
            return False
        checkpoint = torch.load(resume_path)
        #! key variables
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.encoderOptimizer.load_state_dict(checkpoint['Eoptimizer'])
        self.decoderOptimizer.load_state_dict(checkpoint['Doptimizer'])
        print('[*] checkpoint:%d loaded successfully.'%checkpoint['epoch'])
        return True

        
if __name__ == '__main__':
    print("Pytorch version:", torch.__version__)
    print("FLAG: %s" % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', type=str, default='./metaFM_script.json', help='path of configure file')
    parser.add_argument('-r', '--resume_mode', action='store_true', default=False, help='resume from checkpoint or not')
    args = parser.parse_args()
    if args.config_path is not None:
        config_dict = json.load(open(args.config_path))
        node = Trainer(config_dict, resume_mode=args.resume_mode)
        node._train()
    else:
        raise Exception("Unknow --config_path")