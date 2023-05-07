dataset_type = 'PVSGSingleVideoImageDataset'
data_root = './data/pvsg_v1'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False
)
image_size = (360, 480)

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
    test=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',
        test_mode=True,
        pipeline=test_pipeline,
        video_name="0010_8610561401", # for a single video
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
