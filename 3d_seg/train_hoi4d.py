import torch
import torch.nn as nn
import torch.optim as optim
import time, sys, os, random
from tensorboardX import SummaryWriter
import numpy as np

from utils_sailvos3d.config import cfg
from utils_sailvos3d.log import logger
import utils_sailvos3d.utils as utils

from data.hoi4d import Dataset

import torch.distributed as dist  
from torch.utils.data.distributed import DistributedSampler 
# from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.backends.cudnn as cudnn

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,5,6"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def save_best_model(model, optimizer, exp_path, save_name, use_cuda=True):
    f = os.path.join(exp_path, exp_name + save_name)
    logger.info('Saving ' + f)
    model.cpu()
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
        }, f)
    if use_cuda:
        model.cuda()

def init():
    #os.environ['CUDA_VISIBLE_DEVICES']= '1,2,3'
    cudnn.benchmark = False
    
    # copy important files to backup
    backup_dir = os.path.join(cfg.exp_path, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.system('cp train.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(cfg.model_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.config, backup_dir))

    # log the config
    logger.info(cfg)

    # summary writer
    global writer
    writer = SummaryWriter(cfg.exp_path)

    torch.cuda.set_device(0)
    if cfg.distributed:
        dist.init_process_group(backend="nccl")

    if cfg.distributed:
        manual_seed = cfg.manual_seed + cfg.local_rank
    else:
        manual_seed = cfg.manual_seed

    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed_all(cfg.manual_seed)

def train_epoch(trainloader, model, model_fn, optimizer, epoch):
    iter_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    am_dict = {}
    
    model.train()
    start_epoch = time.time()
    end = time.time()
    for i, data in enumerate(trainloader):
        data_time.update(time.time() - end)
        torch.cuda.empty_cache()

        ##### adjust learning rate
        #lr = utils.step_learning_rate(optimizer, cfg.lr, epoch - 1, cfg.step_epoch, cfg.multiplier)
        lr = utils.cosine_lr_after_step(optimizer, cfg.lr, epoch - 1, cfg.step_epoch, cfg.epochs)
        ##### prepare input and forward
        # logger.info('Training sample: {}'.format(data['id']))
        # logger.info('Training sample range: {}'.format(data['range']))
        pred = model_fn(model, data, epoch)
        # try:
        #     pred = model_fn(model, data, epoch)
        # except:
        #     logger.info("An error happens")
        #     torch.cuda.empty_cache()
        #     break
        # else:
        if pred == None: continue
        loss, _, visual_dict, meter_dict = pred
        ##### meter_dict
        for k, v in meter_dict.items():
            if k not in am_dict.keys():
                am_dict[k] = utils.AverageMeter()
            am_dict[k].update(v[0], v[1])
            
        ##### backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ##### time and print
        current_iter = (epoch - 1) * len(trainloader) + i + 1
        max_iter = cfg.epochs * len(trainloader)
        remain_iter = max_iter - current_iter

        iter_time.update(time.time() - end)
        end = time.time()

        remain_time = remain_iter * iter_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (cfg.distributed == False) or (dist.get_rank() == 0):
            sys.stdout.write(
                "epoch: {}/{} iter: {}/{}  lr: {} loss: {:.4f}({:.4f}) data_time: {:.2f}({:.2f}) iter_time: {:.2f}({:.2f}) remain_time: {remain_time}\n".format
                (epoch, cfg.epochs, i + 1, len(trainloader), lr, am_dict['loss'].val, am_dict['loss'].avg,
                data_time.val, data_time.avg, iter_time.val, iter_time.avg, remain_time=remain_time))
            if (i == len(trainloader) - 1): print()
        # sys.stdout.write(
        #     "epoch: {}/{} iter: {}/{}  lr: {} loss: {:.4f}({:.4f}) data_time: {:.2f}({:.2f}) iter_time: {:.2f}({:.2f}) remain_time: {remain_time}\n".format
        #     (epoch, cfg.epochs, i + 1, len(trainloader), lr, am_dict['loss'].val, am_dict['loss'].avg,
        #     data_time.val, data_time.avg, iter_time.val, iter_time.avg, remain_time=remain_time))
        # if (i == len(trainloader) - 1): print()
        # torch.cuda.empty_cache()

    if (cfg.distributed == False) or (dist.get_rank() == 0):
        logger.info("epoch: {}/{}, train loss: {:.4f}, time: {}s".format(epoch, cfg.epochs, am_dict['loss'].avg, time.time() - start_epoch))
    # logger.info("epoch: {}/{}, train loss: {:.4f}, time: {}s".format(epoch, cfg.epochs, am_dict['loss'].avg, time.time() - start_epoch))
	
    if (cfg.distributed == False) or (dist.get_rank() == 0):
        utils.checkpoint_save(model, optimizer, cfg.exp_path, cfg.config.split('/')[-1][:-5], epoch, cfg.save_freq, use_cuda)    
    
    for k in am_dict.keys():
        if k in visual_dict.keys():
            writer.add_scalar(k+'_train', am_dict[k].avg, epoch)

def eval_epoch(val_loader, model, model_fn, epoch):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    am_dict = {}

    with torch.no_grad():
        model.eval()
        start_epoch = time.time()
        for i, batch in enumerate(val_loader):

            ##### prepare input and forward
            loss, preds, visual_dict, meter_dict = model_fn(model, batch, epoch)

            ##### meter_dict
            for k, v in meter_dict.items():
                if k not in am_dict.keys():
                    am_dict[k] = utils.AverageMeter()
                am_dict[k].update(v[0], v[1])

            ##### print
            sys.stdout.write("\riter: {}/{} loss: {:.4f}({:.4f})".format(i + 1, len(val_loader), am_dict['loss'].val, am_dict['loss'].avg))
            if (i == len(val_loader) - 1): print()

        logger.info("epoch: {}/{}, val loss: {:.4f}, time: {}s".format(epoch, cfg.epochs, am_dict['loss'].avg, time.time() - start_epoch))

        for k in am_dict.keys():
            if k in visual_dict.keys():
                writer.add_scalar(k + '_eval', am_dict[k].avg, epoch)

if __name__ == '__main__':
    #### init
    init()
    if cfg.distributed:
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    ##### get model version and data version
    exp_name = cfg.config.split('/')[-1][:-5]
    model_name = exp_name.split('_')[0]
    data_name = exp_name.split('_')[-1]

    ##### model
    logger.info('=> creating model ...')
    
    from model.DKNet import DKNet
    from model.DKNet import train_fn, test_fn
    from test_epoch_hoi4d import test_epoch
    
    model = DKNet(cfg)
    ##### model_fn
    model_fn = train_fn
    model_test_fn = test_fn

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.to(device)
    # os.system("watch nvidia-smi")

    if cfg.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        model = model.cuda()

    # logger.info(model)
    logger.info('#classifier parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    ##### optimizer
    if cfg.optim == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    elif cfg.optim == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    elif cfg.optim == 'AdamW':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)

    ##### datasets
    if cfg.dataset == 'hoi4d':
        dataset = Dataset(test=False)
        dataset.trainLoader()
        dataset.testLoader()
        train_dataloader = dataset.train_data_loader
        test_dataloader = dataset.test_data_loader
    
    ##### resume
    start_epoch = utils.checkpoint_restore(model, optimizer, cfg.exp_path, cfg.config.split('/')[-1][:-5], use_cuda, dist = cfg.distributed, epoch = cfg.start_epoch)      # resume from the latest epoch, or specify the epoch to restore

    ##### train and val
    best_ap, best_ap_50, best_ap_25 = utils.ap_restore(cfg.exp_path, exp_name)
    if (cfg.distributed == False) or (dist.get_rank() == 0):
        logger.info('cuda available: {}'.format(use_cuda))
        logger.info('#classifier parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))
        logger.info("***Best model with AP %.2f, AP50 %.2f and AP25 %.2f.***"%(best_ap, best_ap_50, best_ap_25))
    # logger.info("***Best model with AP %.2f, AP50 %.2f and AP25 %.2f.***"%(best_ap, best_ap_50, best_ap_25))
    
    for epoch in range(start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            dataset.train_data_loader.sampler.set_epoch(epoch)
            # dataset.val_data_loader.sampler.set_epoch(epoch)

        torch.cuda.empty_cache()
        train_epoch(train_dataloader, model, model_fn, optimizer, epoch)
        dataset.trainLoader()
        train_dataloader = dataset.train_data_loader
        
        if (utils.is_multiple(epoch, cfg.save_freq) or utils.is_power2(epoch)) and epoch > (cfg.prepare_epochs + cfg.semantic_epochs):
            if (cfg.distributed == False) or (dist.get_rank() == 0):
                ap_list = test_epoch(model, model_test_fn, test_dataloader, dataset, epoch, logger, cfg)
                if best_ap < ap_list[0]:
                    best_ap = ap_list[0]
                    save_best_model(model, optimizer, cfg.exp_path, 'ap_best_model_{:.0f}.pth'.format(best_ap*100), use_cuda=True)
                    logger.info("***Saving model with AP %.2f, AP50 %.2f and AP25 %.2f on epoch %d.***"%(best_ap, ap_list[1], ap_list[2], epoch))
                    
                if best_ap_50 < ap_list[1]:
                    best_ap_50 = ap_list[1]
                    save_best_model(model, optimizer, cfg.exp_path, 'ap50_best_model_{:.0f}.pth'.format(best_ap_50*100), use_cuda=True)
                    logger.info("***Saving model with AP %.2f, AP50 %.2f and AP25 %.2f on epoch %d.***"%(best_ap, ap_list[1], ap_list[2], epoch))
                
                if best_ap_25 < ap_list[2]:
                    best_ap_25 = ap_list[2]
                    save_best_model(model, optimizer, cfg.exp_path, 'ap25_best_mode_{:.0f}.pth'.format(best_ap_25*100), use_cuda=True)
                    logger.info("***Saving model with AP %.2f, AP50 %.2f and AP25 %.2f on epoch %d.***"%(best_ap, ap_list[1], ap_list[2], epoch))