dataset_type = 'PVSGImageDataset'
data_root = './data/pvsg_v1'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False
)
image_size = (360, 480)
# The kitti dataset contains 1226 x 370 and 1241 x 376
train_pipeline = [
    dict(type='LoadImgDirect', with_depth=True),
    dict(type='LoadAnnotationsDirect'),
    dict(type='Resize', img_scale=image_size, keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomCrop', crop_size=image_size, crop_type='absolute', recompute_bbox=True, allow_negative_crop=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImgDirect'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(360, 480),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            split='train',
            test_mode=False,
            pipeline=train_pipeline,
        )
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',
        test_mode=True,
        pipeline=test_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

evaluation = dict(interval=1, metric=['PQ'])
