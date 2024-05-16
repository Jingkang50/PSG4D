import torch
import time
import numpy as np
import random
import os
import scipy.stats as stats

from utils_sailvos3d.config import cfg
cfg.task = 'test'
from utils_sailvos3d.log import logger
import utils_sailvos3d.utils as utils
import utils_sailvos3d.eval_hoi4d as eval
import torch.distributed as dist  

# from data.scannetv2_inst import Dataset
# from data.sailvos3d_4 import Dataset
from data.hoi4d import Dataset

CLASS_LABELS = ['no label',
'noise',
'Chair',
'table',
'water dispenser',
'ground',
'void',
'void',
'void',
'bucket',
'garbage can',
'wall',
'sofa',
'bag',
'void',
'bowl',
'bottled drinks',
'mug',
'void',
'locker',
'express box',
'kettle',
'hammer',
'pliers',
'toy car',
'laptop',
'refrigerator',
'safe',
'stapler',
'desk lamp',
'trolley case',
'Knife',
'cupboard',
'bed',
'bedside table',
'Door',
'window',
'mural',
'counter',
'curtain',
'pool',
'Pillow',
'carpet',
'television',
'ceiling',
'other furniture',
'other objects',
'void',
'Scissors']

def init():
    global result_dir
    result_dir = os.path.join(cfg.exp_path, 'result', 'epoch{}_scoret{}_npointt{}'.format(cfg.test_epoch, cfg.TEST_SCORE_THRESH, cfg.TEST_NPOINT_THRESH), cfg.split)
    backup_dir = os.path.join(result_dir, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'predicted_masks'), exist_ok=True)
    os.system('cp test.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(cfg.model_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.config, backup_dir))
    
    if cfg.distributed:
        dist.init_process_group(backend="nccl")

    global semantic_label_idx
    # semantic_label_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
    semantic_label_idx = [i for i in range(cfg.classes)]

    logger.info(cfg)

    random.seed(cfg.test_seed)
    np.random.seed(cfg.test_seed)
    torch.manual_seed(cfg.test_seed)
    torch.cuda.manual_seed_all(cfg.test_seed)

def test(model, model_fn, data_name, epoch):
    logger.info('>>>>>>>>>>>>>>>> Start Testing >>>>>>>>>>>>>>>>')
    #### datasets
    dataset = Dataset(test=True)
    dataset.testLoader()
    if cfg.distributed:
        dataset.test_data_loader.sampler.set_epoch(epoch)
    dataloader = dataset.test_data_loader


    with torch.no_grad():
        model.eval()
        start = time.time()

        matches = {}
        times = 0

        for i, data in enumerate(dataloader):
            torch.cuda.empty_cache()
            start1 = time.time()
            preds = model_fn(data, model, epoch)

            if preds == None:
                break

            coords = data['locs'].cuda()
            superpoint = data['superpoint']
            ##### get semantic predictions
            semantic_scores = preds['semantic']
            semantic_preds = semantic_scores.max(1)[1]

            pt_offsets = preds['pt_offsets']
            merged_pt= preds['merge'][0]

            ##### get offsets predictions
            candidate_pt, candidate_score = preds['candidate']

            if (epoch > cfg.prepare_epochs):
                masks, thres, thre_masks, seg_result, ins_sem_preds, weights, kernel_merge = preds['decoding']

                ### post-processing
                # cover score
                org_num = thre_masks.sum(1) + 1e-6
                masks_score = torch.zeros(masks.shape[0]).cuda().long() # (I, N), int, cuda
                masks_num = torch.zeros(masks.shape[0]).cuda().float()
                for i in range(masks_score.shape[0]): masks_score[i] = masks[i, seg_result == i].sum()
                cover_scores = torch.clamp((masks_score.float() / org_num), max=1.0)

                thre_masks = torch.sqrt(thre_masks * cover_scores.unsqueeze(-1))
                thres = torch.sqrt(thres * cover_scores)
                seg_score, seg_result = thre_masks.max(0)
                if torch.sum(thres != 0):
                    seg_result[seg_score < thres[thres != 0].min()] = -100

                # mask number
                for i in range(masks_score.shape[0]): masks_num[i] = (seg_result == i).sum()
                num_mask = (masks_num > cfg.TEST_NPOINT_THRESH)

                if torch.sum(num_mask):
                    merged_pt = merged_pt[num_mask]
                    ins_sem_preds = ins_sem_preds[num_mask]
                    cover_scores = cover_scores[num_mask]
                    thres = thres[num_mask]
                    thre_masks = thre_masks[num_mask]
                    weights = weights[num_mask]
                    kernel_merge = kernel_merge[num_mask]

                    seg_score, seg_result = thre_masks.max(0)
                    if torch.sum(thres != 0):
                        seg_result[seg_score < thres[thres != 0].min()] = -100

                # prediction scores
                semantic_scores_soft = semantic_scores.softmax(-1)
                scores = torch.zeros_like(cover_scores)
                for i in range(num_mask.sum()):
                    pts_num = (seg_result == i).sum()
                    scores[i] = thre_masks[i, seg_result == i].sum() / (pts_num + 1e-6)
                    if ins_sem_preds[i] == cfg.classes: scores[i] = 0
                    else: scores[i] *= semantic_scores_soft[(seg_result == i), ins_sem_preds[i].long()].sum() / (pts_num + 1e-6)
                scores = torch.sqrt(scores)

                score_mask = scores > cfg.TEST_SCORE_THRESH
                if torch.sum(score_mask):
                    merged_pt = merged_pt[score_mask]
                    scores = scores[score_mask]
                    thres = thres[score_mask]
                    thre_masks = thre_masks[score_mask]
                    weights = weights[score_mask]
                    kernel_merge = kernel_merge[score_mask]

                    seg_score, seg_result = thre_masks.max(0)
                    if torch.sum(thres != 0):
                        seg_result[seg_score < thres[thres != 0].min()] = -100

                # superpoint refinement
                # superpoint = torch.unique(superpoint, return_inverse=True)[1]

                # sp_labels, sp_scores = utils.align_superpoint_label(seg_result, superpoint, thre_masks.shape[0])
                # seg_result_refine = sp_labels[superpoint]

                # seg_result = seg_result_refine

                object_idxs = torch.nonzero(torch.logical_and(torch.logical_and(semantic_preds != 5, semantic_preds != 11), torch.logical_and(semantic_preds != 0, semantic_preds != 1))).view(-1)

                ins_sem_preds = utils.get_instance_seg_pred_label(semantic_preds[object_idxs], seg_result[object_idxs], thre_masks.shape[0])

                batch_idxs = coords[:, 0]
                ins_batch_idxs = batch_idxs[merged_pt]


                end1 = time.time() - start1

                for idx in range(batch_idxs.max()+1):
                    test_scene_name = dataset.test_file_names[int(data['id'][idx])]
                    # test_scene_name = dataset.images_path[int(data['id'][idx])].split('/')[-3] + dataset.images_path[int(data['id'][idx])].split('/')[-1][:-4]
                    ins_batch_mask = (ins_batch_idxs == idx)
                    pts_batch_mask = (batch_idxs == idx)
                    masks_batch = thre_masks[ins_batch_mask]
                    masks_category_batch = ins_sem_preds[ins_batch_mask]
                    scores_pred_batch = scores[ins_batch_mask]
                    weights_pred_batch = weights[ins_batch_mask]
                    kernel_merge_pred_batch = kernel_merge[ins_batch_mask]

                    N = pts_batch_mask.sum()
                    masks_pred = torch.zeros((masks_batch.shape[0], N), dtype=torch.int, device=masks.device) # (nProposal, N), int, cuda
                    for ii, ins_idx in enumerate(torch.where(ins_batch_mask)[0]): masks_pred[ii] = (seg_result[pts_batch_mask] == ins_idx).int()

                    print('invalid ins:{}'.format((masks_category_batch == cfg.classes).sum()))
                    cat_mask = (masks_category_batch != cfg.classes)
                    scores_pred_batch = scores_pred_batch[cat_mask]
                    masks_pred = masks_pred[cat_mask]
                    masks_category_batch = masks_category_batch[cat_mask]
                    weights_pred_batch = weights_pred_batch[cat_mask]
                    kernel_merge_pred_batch = kernel_merge_pred_batch[cat_mask]

                    semantic_id = torch.tensor(semantic_label_idx, device=masks_category_batch.device)[masks_category_batch.long()]

                    clusters = masks_pred
                    cluster_scores = scores_pred_batch
                    cluster_semantic_id = semantic_id

                    nclusters = clusters.shape[0]

                    ##### prepare for evaluation
                    if cfg.eval:
                        pred_info = {}
                        pred_info['conf'] = cluster_scores.cpu().numpy()
                        pred_info['label_id'] = cluster_semantic_id.cpu().numpy()
                        pred_info['mask'] = clusters.cpu().numpy()
                        pred_info['weights'] = weights_pred_batch.cpu().numpy()
                        pred_info['kernel_merge'] = kernel_merge_pred_batch.cpu().numpy()
                        # gt_file = os.path.join(cfg.data_root, cfg.dataset, cfg.split + '_gt', test_scene_name + '.txt')
                        gt2pred, pred2gt = eval.assign_instances_for_scan(test_scene_name, pred_info, data['labels'], data['instance_labels'])
                        matches[test_scene_name] = {}
                        matches[test_scene_name]['gt'] = gt2pred
                        matches[test_scene_name]['pred'] = pred2gt

                        if cfg.split == 'val':
                            matches[test_scene_name]['seg_gt'] = data['labels'][pts_batch_mask]
                            matches[test_scene_name]['seg_pred'] = semantic_preds[pts_batch_mask]
                            # print(np.unique(data['labels'][pts_batch_mask], return_counts=True)[0], np.unique(data['labels'][pts_batch_mask], return_counts=True)[1])
                            # print(np.unique(semantic_preds[pts_batch_mask].cpu().numpy(), return_counts=True)[0], np.unique(semantic_preds[pts_batch_mask].cpu().numpy(), return_counts=True)[1])
                    if cfg.save_res:
                        os.makedirs(os.path.join(result_dir, 'segmentation_results', test_scene_name[40:test_scene_name.rfind('/')]), exist_ok=True)
                        # print(os.path.join(result_dir, 'segmentation_results', test_scene_name[40:test_scene_name.rfind('/')]))
                        semantic_np = semantic_preds[pts_batch_mask].cpu().numpy()
                        print("Saving to ", os.path.join(result_dir, 'segmentation_results', test_scene_name[40:]))
                        np.savez(os.path.join(result_dir, 'segmentation_results', test_scene_name[40:]), semantic_res = semantic_np,
                        instance_conf = pred_info['conf'], instance_label = pred_info['label_id'], instance_mask = pred_info['mask'], weights = pred_info['weights'], kernel_merge = pred_info['kernel_merge'])
                    ##### save files
                    start3 = time.time()
                    if cfg.save_semantic:
                        os.makedirs(os.path.join(result_dir, 'semantic'), exist_ok=True)
                        semantic_np = semantic_preds[pts_batch_mask].cpu().numpy()
                        np.save(os.path.join(result_dir, 'semantic', test_scene_name + '.npy'), semantic_np)

                    if cfg.save_pt_offsets:
                        os.makedirs(os.path.join(result_dir, 'coords_offsets'), exist_ok=True)
                        pt_offsets_np = pt_offsets[pts_batch_mask].cpu().numpy()
                        coords_np = data['locs_float'][pts_batch_mask].numpy()
                        coords_offsets = np.concatenate((coords_np, pt_offsets_np), 1)   # (N, 6)
                        np.save(os.path.join(result_dir, 'coords_offsets', test_scene_name + '.npy'), coords_offsets)

                    if cfg.save_candidate:
                        os.makedirs(os.path.join(result_dir, 'candidate'), exist_ok=True)
                        candidate_pt_cpu = (candidate_pt[pts_batch_mask[candidate_pt.long()]] - torch.where(pts_batch_mask)[0].min()).cpu().numpy()
                        merged_pt_cpu = (merged_pt[pts_batch_mask[merged_pt.long()]] - torch.where(pts_batch_mask)[0].min()).cpu().numpy()
                        candidate_score_ = torch.zeros_like(semantic_preds).cuda().float()
                        # candidate_score_[semantic_preds>=cfg.invalid_classes] = candidate_score
                        candidate_score_[semantic_preds<=(cfg.classes-2)] = candidate_score
                        candidate_score_cpu = candidate_score_[pts_batch_mask].cpu().numpy()
                        np.save(os.path.join(result_dir, 'candidate', test_scene_name + '.npy'), candidate_pt_cpu)
                        np.save(os.path.join(result_dir, 'candidate', test_scene_name + '_merge.npy'), merged_pt_cpu)
                        np.save(os.path.join(result_dir, 'candidate', test_scene_name + '_score.npy'), candidate_score_cpu)

                    if(epoch > cfg.prepare_epochs and cfg.save_instance):
                        f = open(os.path.join(result_dir, test_scene_name + '.txt'), 'w')
                        for proposal_id in range(nclusters):
                            clusters_i = clusters[proposal_id].cpu().numpy()  # (N)
                            semantic_label = masks_category_batch[proposal_id].long().cpu()
                            score = cluster_scores[proposal_id]
                            f.write('predicted_masks/{}_{:03d}.txt {} {:.4f}'.format(test_scene_name, proposal_id, semantic_label_idx[semantic_label], score))
                            if proposal_id < nclusters - 1:
                                f.write('\n')
                            np.savetxt(os.path.join(result_dir, 'predicted_masks', test_scene_name + '_%03d.txt' % (proposal_id)), clusters_i, fmt='%d')
                        f.close()

                    end3 = time.time() - start3
                    end = time.time() - start
                    start = time.time()

                    ##### print
                    logger.info("instance iter: {}/{} point_num: {} ncluster: {} time: total {:.2f}s inference {:.2f}s save {:.2f}s".format(data['id'][idx] + 1, len(dataset.test_files), N, nclusters, end, end1, end3))
                    times += end1

        ##### evaluation
        if cfg.eval:
            ap_scores = eval.evaluate_matches(matches)
            #np.save(os.path.join(result_dir, 'ap_scores.npy'), ap_scores)
            avgs = eval.compute_averages(ap_scores)
            eval.print_results(avgs)

        # evaluate semantic segmantation accuracy and mIoU
        if cfg.split == 'val':
            seg_accuracy = evaluate_semantic_segmantation_accuracy(matches)
            logger.info("semantic_segmantation_accuracy: {:.4f}".format(seg_accuracy))
            miou = evaluate_semantic_segmantation_miou(matches)
            logger.info("semantic_segmantation_mIoU: {:.4f}".format(miou))

        scan_nums = (len(dataset.test_files) * cfg.test_batch_size)
        average_time = times / scan_nums

        logger.info("averge inference time: {:.4f}".format(average_time))


def evaluate_semantic_segmantation_accuracy(matches):
    seg_gt_list = []
    seg_pred_list = []
    for k, v in matches.items():
        seg_gt_list.append(v['seg_gt'].cpu())
        seg_pred_list.append(v['seg_pred'].cpu())
    # seg_gt_all = torch.cat(seg_gt_list, dim=0).cuda()
    # seg_pred_all = torch.cat(seg_pred_list, dim=0).cuda()
    seg_gt_all = torch.cat(seg_gt_list, dim=0)
    seg_pred_all = torch.cat(seg_pred_list, dim=0)
    assert seg_gt_all.shape == seg_pred_all.shape
    correct = (seg_gt_all[seg_gt_all != -100] == seg_pred_all[seg_gt_all != -100]).sum()
    whole = (seg_gt_all != -100).sum()
    seg_accuracy = correct.float() / whole.float()
    return seg_accuracy

def evaluate_semantic_segmantation_miou(matches):
    seg_gt_list = []
    seg_pred_list = []
    points_list = []
    for k, v in matches.items():
        seg_gt_list.append(v['seg_gt'].cpu())
        seg_pred_list.append(v['seg_pred'].cpu())
    # seg_gt_all = torch.cat(seg_gt_list, dim=0).cuda()
    # seg_pred_all = torch.cat(seg_pred_list, dim=0).cuda()
    seg_gt_all = torch.cat(seg_gt_list, dim=0)
    seg_pred_all = torch.cat(seg_pred_list, dim=0)
    assert seg_gt_all.shape == seg_pred_all.shape
    iou_list = []
    cls_list = []
    for _index in seg_gt_all.unique():
        if _index != -100:
            intersection = ((seg_gt_all == _index) &  (seg_pred_all == _index)).sum()
            union = ((seg_gt_all == _index) | (seg_pred_all == _index)).sum()
            iou = intersection.float() / union
            cls_list.append(CLASS_LABELS[_index])
            iou_list.append(iou)
            points_list.append((seg_gt_all == _index).sum())
    print(len(iou_list))
    print("IoU list: ")
    for i in range(len(CLASS_LABELS)):
        if CLASS_LABELS[i] in cls_list:
            print(CLASS_LABELS[i], iou_list[cls_list.index(CLASS_LABELS[i])])
        else:
            print(CLASS_LABELS[i], "None")
    print("Point number: ")
    for i in range(len(CLASS_LABELS)):
        if CLASS_LABELS[i] in cls_list:
            print(CLASS_LABELS[i], points_list[cls_list.index(CLASS_LABELS[i])])
        else:
            print(CLASS_LABELS[i], "None")
    iou_tensor = torch.tensor(iou_list)
    miou = iou_tensor.mean()
    return miou


if __name__ == '__main__':
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
    logger.info('Classes: {}'.format(cfg.classes))

    from model.DKNet import DKNet
    from model.DKNet import test_fn
    model = DKNet(cfg)
    ##### model_fn
    model_fn = test_fn

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()
    
    if cfg.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    # logger.info(model)
    logger.info('#classifier parameters (model): {}'.format(sum([x.nelement() for x in model.parameters()])))

    ##### model_fn (criterion)
    model_fn = test_fn

    ##### load model
    start_epoch = utils.checkpoint_restore(model, None, cfg.exp_path, cfg.config.split('/')[-1][:-5], use_cuda, cfg.test_epoch, dist=cfg.distributed, f=cfg.pretrain)      # resume from the latest epoch, or specify the epoch to restore

    ##### evaluate
    test(model, model_fn, data_name, start_epoch)
