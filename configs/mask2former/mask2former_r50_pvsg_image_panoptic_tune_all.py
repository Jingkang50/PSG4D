_base_ = ['./mask2former_r50_lsj_8x2_50e_coco-panoptic_custom.py']

# tune all with lr * 0.1

# load mask2former coco r50
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/' \
            'mask2former/mask2former_r50_lsj_8x2_50e_coco-panoptic/' \
            'mask2former_r50_lsj_8x2_50e_coco-panoptic_20220326_224516-11a44721.pth'


runner = dict(type='EpochBasedRunner', max_epochs=8)


embed_multi = dict(lr_mult=1.0, decay_mult=0.0)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.00005,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=False,
    step=[327778, 355092],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=1.0,  # no warmup
    warmup_iters=10)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook', by_epoch=False)
    ])
interval = 1
workflow = [('train', interval)]
checkpoint_config = dict(
    by_epoch=True, interval=interval, save_last=True)

# Before 365001th iteration, we do evaluation every 5000 iterations.
# After 365000th iteration, we do evaluation every 368750 iterations,
# which means that we do evaluation at the end of training.
# dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
evaluation = dict(
    interval=5000000000, # no eval
    metric=['PQ'])