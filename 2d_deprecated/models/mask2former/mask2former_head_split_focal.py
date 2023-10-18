# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, build_plugin_layer, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.ops import point_sample
from mmcv.runner import ModuleList, force_fp32

from mmdet.core import build_assigner, build_sampler, reduce_mean, multi_apply
from mmdet.models.utils import get_uncertain_point_coords_with_randomness
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.dense_heads.maskformer_head import MaskFormerHead


@HEADS.register_module()
class Mask2FormerHeadSplitFocal(MaskFormerHead):
    """Implements the Mask2Former head.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer decoder.
        pixel_decoder (:obj:`mmcv.ConfigDict` | dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of tranformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder position encoding. Defaults to None.
        loss_cls (:obj:`mmcv.ConfigDict` | dict): Config of the classification
            loss. Defaults to None.
        loss_mask (:obj:`mmcv.ConfigDict` | dict): Config of the mask loss.
            Defaults to None.
        loss_dice (:obj:`mmcv.ConfigDict` | dict): Config of the dice loss.
            Defaults to None.
        train_cfg (:obj:`mmcv.ConfigDict` | dict): Training config of
            Mask2Former head.
        test_cfg (:obj:`mmcv.ConfigDict` | dict): Testing config of
            Mask2Former head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 num_things_classes=80,
                 num_stuff_classes=53,
                 num_queries=100,
                 num_transformer_feat_level=3,
                 pixel_decoder=None,
                 enforce_decoder_input_project=False,
                 transformer_decoder=None,
                 positional_encoding=None,
                 thing_class_weight=None,
                 stuff_class_weight=None,
                 loss_cls=None,
                 loss_mask=None,
                 loss_dice=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.transformerlayers.\
            attn_cfgs.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        assert pixel_decoder.encoder.transformerlayers.\
            attn_cfgs.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels)
        self.pixel_decoder = build_plugin_layer(pixel_decoder_)[1]
        self.transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv2d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_positional_encoding = build_positional_encoding(
            positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level,
                                        feat_channels)
        # cls embed output
        self.thing_embed = nn.Linear(feat_channels, self.num_things_classes)
        self.stuff_embed = nn.Linear(feat_channels, self.num_stuff_classes)

        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        if train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            self.sampler = build_sampler(self.train_cfg.sampler, context=self)
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.thing_class_weight = thing_class_weight
        self.stuff_class_weight = stuff_class_weight

        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)

    def init_weights(self):
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks,
                           img_metas):
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, h, w).
            gt_labels (Tensor): Ground truth class indices for one image with
                shape (num_gts, ).
            gt_masks (Tensor): Ground truth mask for each image, each with
                shape (num_gts, h, w).
            img_metas (dict): Image information.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image. \
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image. \
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image. \
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image. \
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each \
                    image.
                - neg_inds (Tensor): Sampled negative indices for each \
                    image.
        """

        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)

        # shape (num_queries, num_points)
        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1,
                                                        1)).squeeze(1)
        # shape (num_gts, num_points)
        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1,
                                                               1)).squeeze(1)

        # assign and sample
        assign_result = self.assigner.assign(cls_score, mask_points_pred,
                                             gt_labels, gt_points_masks,
                                             img_metas)
        sampling_result = self.sampler.sample(assign_result, mask_pred,
                                              gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        if num_queries > self.num_stuff_classes:
            label_default = self.num_things_classes
        else:
            label_default = self.num_stuff_classes

        # label target (thing or stuff labels)
        labels = gt_labels.new_full((num_queries, ),
                                    label_default,
                                    dtype=torch.long)

        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((num_queries, ))

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((num_queries, ))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds)

    def loss_single(self, cls_scores, mask_preds, gt_labels_list,
                    gt_masks_list, img_metas, is_thing_list):
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image, each with shape (num_gts, ).
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (num_gts, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         num_total_pos,
         num_total_neg) = self.get_targets(cls_scores_list, mask_preds_list,
                                           gt_labels_list, gt_masks_list,
                                           img_metas)

        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classification loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        # class_weight = cls_scores.new_tensor(self.class_weight)
        if is_thing_list:
            class_weight = cls_scores.new_tensor(self.thing_class_weight)
        else:
            class_weight = cls_scores.new_tensor(self.stuff_class_weight)

        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum()
        )

        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)

        return loss_cls, loss_mask, loss_dice

    def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list,
                    gt_masks_list, img_metas):
        """Compute classification and mask targets for all images for a decoder
        layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape (num_queries,
                cls_out_channels).
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape (num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for all
                images. Each with shape (n, ), n is the sum of number of stuff
                type and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[list[Tensor]]: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels of all images.\
                    Each with shape (num_queries, ).
                - label_weights_list (list[Tensor]): Label weights\
                    of all images. Each with shape (num_queries, ).
                - mask_targets_list (list[Tensor]): Mask targets of\
                    all images. Each with shape (num_queries, h, w).
                - mask_weights_list (list[Tensor]): Mask weights of\
                    all images. Each with shape (num_queries, ).
                - num_total_pos (int): Number of positive samples in\
                    all images.
                - num_total_neg (int): Number of negative samples in\
                    all images.
        """
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list,
                                      mask_preds_list, gt_labels_list,
                                      gt_masks_list, img_metas)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, mask_targets_list,
                mask_weights_list, num_total_pos, num_total_neg)

    def forward_head(self, decoder_out, mask_feature, attn_mask_target_size):
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (num_queries, batch_size, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

            - cls_pred (Tensor): Classification scores in shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred (Tensor): Mask scores in shape \
                (batch_size, num_queries,h, w).
            - attn_mask (Tensor): Attention mask in shape \
                (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)
        # shape (num_queries, batch_size, c) -> (b, n, c)
        # cls_pred = self.cls_embed(decoder_out)

        th_cls_pred = self.thing_embed(decoder_out[:, :-self.num_stuff_classes])
        st_cls_pred = self.stuff_embed(decoder_out[:, -self.num_stuff_classes:])

        # (N_{th}, b, c_{th}) (N_{st}, b, c_{stuff})
        cls_pred = [th_cls_pred, st_cls_pred]

        # shape (num_queries, batch_size, c)
        mask_embed = self.mask_embed(decoder_out)

        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,bchw -> bqhw', mask_embed, mask_feature)
        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)

        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5

        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, attn_mask

    def forward(self, feats, img_metas):
        """Forward function.

        Args:
            feats (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two elements.

            - cls_pred_list (list[Tensor)]: Classification logits \
                for each decoder layer. Each is a 3D-tensor with shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred_list (list[Tensor]): Mask logits for each \
                decoder layer. Each with shape (batch_size, num_queries, \
                 h, w).
        """
        batch_size = len(img_metas)
        mask_features, multi_scale_memorys = self.pixel_decoder(feats)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(2, 0, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (num_queries, batch_size, c)
        query_feat = self.query_feat.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self.forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(
                attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            attn_masks = [attn_mask, None]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self.forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list

    def forward_train(self,
                      feats,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_semantic_seg,
                      gt_bboxes_ignore=None):
        """Forward function for training mode.

        Args:
            feats (list[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            img_metas (list[Dict]): List of image information.
            gt_bboxes (list[Tensor]): Each element is ground truth bboxes of
                the image, shape (num_gts, 4). Not used here.
            gt_labels (list[Tensor]): Each element is ground truth labels of
                each box, shape (num_gts,).
            gt_masks (list[BitmapMasks]): Each element is masks of instances
                of a image, shape (num_gts, h, w).
            gt_semantic_seg (list[tensor] | None): Each element is the ground
                truth of semantic segmentation with the shape (N, H, W).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.
            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be
                ignored. Defaults to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # not consider ignoring bboxes
        assert gt_bboxes_ignore is None

        # forward
        all_cls_scores, all_mask_preds = self(feats, img_metas)

        # preprocess ground truth
        gt_labels, gt_masks = self.preprocess_gt(gt_labels, gt_masks,
                                                 gt_semantic_seg, img_metas)

        # loss
        losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks,
                           gt_semantic_seg, img_metas)

        return losses

    @force_fp32(apply_to=('all_cls_scores', 'all_mask_preds'))
    def loss(self, all_cls_scores, all_mask_preds, gt_labels_list,
             gt_masks_list, gt_semantic_seg, img_metas):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape (num_decoder, batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape (num_decoder, batch_size, num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (n, ). n is the sum of number of stuff type
                and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image with
                shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        num_dec_layers = len(all_cls_scores)
        stuff_gt_labels = []
        stuff_gt_masks = []
        thing_gt_labels = []
        thing_gt_masks = []

        for gt_labels, gt_masks in zip(gt_labels_list, gt_masks_list):
            num_stuff = sum(torch.unique(gt_labels) >= self.num_things_classes)
            stuff_gt_labels.append(gt_labels[-num_stuff:])
            stuff_gt_masks.append(gt_masks[-num_stuff:])
            thing_gt_labels.append(gt_labels[:-num_stuff])
            thing_gt_masks.append(gt_masks[:-num_stuff])

        stuff_gt_labels_list = [stuff_gt_labels for _ in range(num_dec_layers)]
        stuff_gt_masks_list = [stuff_gt_masks for _ in range(num_dec_layers)]

        thing_gt_labels_list = [thing_gt_labels for _ in range(num_dec_layers)]
        thing_gt_masks_list = [thing_gt_masks for _ in range(num_dec_layers)]

        stuff_scores = [all_cls_score[1] for all_cls_score in all_cls_scores]
        stuff_mask_preds = [all_mask_pred[:, -self.num_stuff_classes:, :, :] for all_mask_pred in all_mask_preds ]

        thing_scores = [all_cls_score[0] for all_cls_score in all_cls_scores]
        thing_mask_preds = [all_mask_pred[:, :-self.num_stuff_classes, :, :] for all_mask_pred in all_mask_preds]

        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        is_thing = [True for _ in range(num_dec_layers) ]
        not_thing = [False for _ in range(num_dec_layers)]

        losses_cls_stuff, losses_mask_stuff, losses_dice_stuff = multi_apply(
            self.loss_single, stuff_scores, stuff_mask_preds,
            stuff_gt_labels_list, stuff_gt_masks_list, img_metas_list, not_thing)

        losses_cls_thing, losses_mask_thing, losses_dice_thing = multi_apply(
            self.loss_single, thing_scores, thing_mask_preds,
            thing_gt_labels_list, thing_gt_masks_list, img_metas_list, is_thing)

        loss_dict = dict()

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls_stuff[-1] + losses_cls_thing[-1]
        loss_dict['loss_mask'] = losses_mask_stuff[-1] + losses_mask_thing[-1]
        loss_dict['loss_dice'] = losses_dice_stuff[-1] + losses_dice_thing[-1]

        # loss from other decoder layers stuff
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i in zip(
                losses_cls_stuff[:-1],
                losses_mask_stuff[:-1],
                losses_dice_stuff[:-1]):

            loss_dict[f'd{num_dec_layer}.stuff_loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.stuff_loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.stuff_loss_dice'] = loss_dice_i
            num_dec_layer += 1

        # loss from other decoder layers thing
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i in zip(
                losses_mask_thing[:-1],
                losses_mask_thing[:-1],
                losses_mask_thing[:-1]):
            loss_dict[f'd{num_dec_layer}.thing_loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.thing_loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.thing_loss_dice'] = loss_dice_i
            num_dec_layer += 1

        return loss_dict


    def preprocess_gt(self, gt_labels_list, gt_masks_list, gt_semantic_segs,
                      img_metas):
        """Preprocess the ground truth for all images.

        Args:
            gt_labels_list (list[Tensor]): Each is ground truth
                labels of each bbox, with shape (num_gts, ).
            gt_masks_list (list[BitmapMasks]): Each is ground truth
                masks of each instances of a image, shape
                (num_gts, h, w).
            gt_semantic_seg (Tensor | None): Ground truth of semantic
                segmentation with the shape (batch_size, n, h, w).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple: a tuple containing the following targets.
                - labels (list[Tensor]): Ground truth class indices\
                    for all images. Each with shape (n, ), n is the sum of\
                    number of stuff type and number of instance in a image.
                - masks (list[Tensor]): Ground truth mask for each\
                    image, each with shape (n, h, w).
        """
        num_things_list = [self.num_things_classes] * len(gt_labels_list)
        num_stuff_list = [self.num_stuff_classes] * len(gt_labels_list)
        if gt_semantic_segs is None:
            gt_semantic_segs = [None] * len(gt_labels_list)

        targets = multi_apply(preprocess_panoptic_gt_split_th_st, gt_labels_list,
                              gt_masks_list, gt_semantic_segs, num_things_list,
                              num_stuff_list, img_metas)
        labels, masks = targets
        return labels, masks


def preprocess_panoptic_gt_split_th_st(gt_labels, gt_masks, gt_semantic_seg, num_things,
                           num_stuff, img_metas):
    """Preprocess the ground truth for a image.

    Args:
        gt_labels (Tensor): Ground truth labels of each bbox,
            with shape (num_gts, ).
        gt_masks (BitmapMasks): Ground truth masks of each instances
            of a image, shape (num_gts, h, w).
        gt_semantic_seg (Tensor | None): Ground truth of semantic
            segmentation with the shape (1, h, w).
            [0, num_thing_class - 1] means things,
            [num_thing_class, num_class-1] means stuff,
            255 means VOID. It's None when training instance segmentation.
        img_metas (dict): List of image meta information.

    Returns:
        tuple: a tuple containing the following targets.

            - labels (Tensor): Ground truth class indices for a
                image, with shape (n, ), n is the sum of number
                of stuff type and number of instance in a image.
            - masks (Tensor): Ground truth mask for a image, with
                shape (n, h, w). Contains stuff and things when training
                panoptic segmentation, and things only when training
                instance segmentation.
    """
    num_classes = num_things + num_stuff

    things_masks = gt_masks.pad(img_metas['pad_shape'][:2], pad_val=0)\
        .to_tensor(dtype=torch.bool, device=gt_labels.device)

    if gt_semantic_seg is None:
        masks = things_masks.long()
        return gt_labels, masks

    things_labels = gt_labels
    gt_semantic_seg = gt_semantic_seg.squeeze(0)

    semantic_labels = torch.unique(
        gt_semantic_seg,
        sorted=False,
        return_inverse=False,
        return_counts=False)

    stuff_masks_list = []
    stuff_labels_list = []
    for label in semantic_labels:
        if label < num_things or label >= num_classes:
            continue
        stuff_mask = gt_semantic_seg == label
        stuff_masks_list.append(stuff_mask)
        # new stuff label: 0 - N_{stuff} -1
        stuff_labels_list.append(label - num_things)

    if len(stuff_masks_list) > 0:
        stuff_masks = torch.stack(stuff_masks_list, dim=0)
        stuff_labels = torch.stack(stuff_labels_list, dim=0)
        labels = torch.cat([things_labels, stuff_labels], dim=0)
        masks = torch.cat([things_masks, stuff_masks], dim=0)
    else:
        labels = things_labels
        masks = things_masks

    masks = masks.long()

    return labels, masks