import torch
import numpy as np
import random
import os
from utils import utils
import utils.eval as eval

def test_epoch(model, model_fn, dataloader, dataset, epoch, logger, cfg):
    global result_dir
    result_dir = os.path.join(cfg.exp_path, 'result', 'epoch{}_scoret{}_npointt{}'.format(cfg.test_epoch, cfg.TEST_SCORE_THRESH, cfg.TEST_NPOINT_THRESH), cfg.split)
    global semantic_label_idx
    semantic_label_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
    
    logger.info('>>>>>>>>>>>>>>>> Start Testing for epoch %d >>>>>>>>>>>>>>>>'%(epoch))

    with torch.no_grad():
        model = model.eval()
        matches = {}

        for i, data in enumerate(dataloader):
            torch.cuda.empty_cache()
            N = data['feats'].shape[0]
            preds = model_fn(data, model, epoch)
            
            coords = data['locs'].cuda()     
            superpoint = data['superpoint']
            ##### get semantic predictions
            semantic_scores = preds['semantic']
            semantic_preds = semantic_scores.max(1)[1]

            merged_pt= preds['merge'][0]

            if (epoch > cfg.prepare_epochs):
                masks, thres, thre_masks, seg_result, ins_sem_preds = preds['decoding']
                
                org_num = thre_masks.sum(1) + 1e-6
                masks_score = torch.zeros(masks.shape[0]).cuda().long() # (nProposal, N), int, cuda
                masks_num = torch.zeros(masks.shape[0]).cuda().float()
                for i in range(masks_score.shape[0]): masks_score[i] = masks[i, seg_result == i].sum()
                cover_scores = torch.clamp((masks_score.float() / org_num), max=1.0)

                thre_masks = torch.sqrt(thre_masks * cover_scores.unsqueeze(-1))
                thres = torch.sqrt(thres * cover_scores)
                seg_score, seg_result = thre_masks.max(0)
                seg_result[seg_score < thres[thres != 0].min()] = -100


                for i in range(masks_score.shape[0]): masks_num[i] = (seg_result == i).sum()
                num_mask = (masks_num > cfg.TEST_NPOINT_THRESH) 

                merged_pt = merged_pt[num_mask]
                ins_sem_preds = ins_sem_preds[num_mask]
                cover_scores = cover_scores[num_mask]
                thres = thres[num_mask]
                thre_masks = thre_masks[num_mask]

                seg_score, seg_result = thre_masks.max(0)
                seg_result[seg_score < thres[thres != 0].min()] = -100

                semantic_scores_soft = semantic_scores.softmax(-1)
                scores = torch.zeros_like(cover_scores)
                for i in range(num_mask.sum()): 
                    pts_num = (seg_result == i).sum()
                    scores[i] = thre_masks[i, seg_result == i].sum() / (pts_num + 1e-6)
                    if ins_sem_preds[i] == 20: scores[i] = 0
                    else: scores[i] *= semantic_scores_soft[(seg_result == i), ins_sem_preds[i].long()].sum() / (pts_num + 1e-6)
                scores = torch.sqrt(scores)

                score_mask = scores > cfg.TEST_SCORE_THRESH
                merged_pt = merged_pt[score_mask]
                scores = scores[score_mask]
                thres = thres[score_mask]
                thre_masks = thre_masks[score_mask]

                seg_score, seg_result = thre_masks.max(0)
                seg_result[seg_score < thres[thres != 0].min()] = -100

                superpoint = torch.unique(superpoint, return_inverse=True)[1]

                sp_labels, sp_scores = utils.align_superpoint_label(seg_result, superpoint, thre_masks.shape[0])
                seg_result_refine = sp_labels[superpoint]
                
                seg_result = seg_result_refine
                
                object_idxs = torch.nonzero(semantic_preds >= cfg.invalid_classes).view(-1)
                ins_sem_preds = utils.get_instance_seg_pred_label(semantic_preds[object_idxs], seg_result[object_idxs], thre_masks.shape[0])
                
                batch_idxs = coords[:, 0]
                ins_batch_idxs = batch_idxs[merged_pt]   
                
                for i, idx in enumerate(data['id']):
                    test_scene_name = dataset.test_file_names[int(idx)].split('/')[-1][:12]
                    ins_batch_mask = (ins_batch_idxs == i)
                    pts_batch_mask = (coords[:, 0] == i)
                    masks_batch = thre_masks[ins_batch_mask]
                    masks_category_batch = ins_sem_preds[ins_batch_mask]
                    scores_pred_batch = scores[ins_batch_mask]

                    N = pts_batch_mask.sum()
                    masks_pred = torch.zeros((masks_batch.shape[0], N), dtype=torch.int, device=masks.device) # (nProposal, N), int, cuda
                    for ii, ins_idx in enumerate(torch.where(ins_batch_mask)[0]): masks_pred[ii] = (seg_result[pts_batch_mask] == ins_idx).int()

                    cat_mask = (masks_category_batch != 20)
                    print('pred ins:{}'.format(cat_mask.sum()))
                    scores_pred_batch = scores_pred_batch[cat_mask]
                    masks_pred = masks_pred[cat_mask]
                    masks_category_batch = masks_category_batch[cat_mask]

                    semantic_id = torch.tensor(semantic_label_idx, device=masks_category_batch.device)[masks_category_batch.long()]
                
                    clusters = masks_pred#[pick_idxs]
                    cluster_scores = scores_pred_batch#[pick_idxs]
                    cluster_semantic_id = semantic_id#[pick_idxs]

                    #print(clusters, cluster_scores, cluster_semantic_id)
                    ##### prepare for evaluation
                    if cfg.eval:
                        pred_info = {}
                        pred_info['conf'] = cluster_scores.cpu().numpy()
                        pred_info['label_id'] = cluster_semantic_id.cpu().numpy()
                        pred_info['mask'] = clusters.cpu().numpy()
                        gt_file = os.path.join(cfg.data_root, cfg.dataset, cfg.split + '_gt', test_scene_name + '.txt')
                        gt2pred, pred2gt = eval.assign_instances_for_scan(test_scene_name, pred_info, gt_file)
                        #print(gt2pred, pred2gt)
                        matches[test_scene_name] = {}
                        matches[test_scene_name]['gt'] = gt2pred
                        matches[test_scene_name]['pred'] = pred2gt

        ap_scores = eval.evaluate_matches(matches)
        avgs = eval.compute_averages(ap_scores)
        #eval.print_results(avgs)
        logger.info('Epoch %d, Validation. AP: %.2f, AP50: %.2f, AP25: %.2f'%(epoch, avgs['all_ap'], avgs['all_ap_50%'], avgs['all_ap_25%']))
        return [avgs['all_ap'], avgs['all_ap_50%'], avgs['all_ap_25%']]



