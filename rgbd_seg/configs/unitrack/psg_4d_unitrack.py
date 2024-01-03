_base_ = [
    '../mask2former/mask2former_dual_r50_rgbd_test.py'
]

tracker_cfg = dict(
    common=dict(
        exp_name="imagenet_resnet50_s3_womotion_timecycle",
        model_type="imagenet50",
        remove_layers=['layer4'],
        im_mean=[0.485, 0.456, 0.406],
        im_std=[0.229, 0.224, 0.225],
        nopadding=False,
        resume='/mnt/lustre/wxpeng/PSG4D/configs/unitrack/timecycle.pth',
        down_factor=8,
        infer2D=True ,
        workers=4,
        gpu_id=0,
        device='cuda'
    ),
    mots=dict(
        obid='COSTA_st',
        mots_root='/mnt/lustre/jkyang/wxpeng/CVPR23/UniTrack/MOTS/MOTS',
        save_videos=True,
        save_images=True,
        test=False,
        track_buffer=30,
        nms_thres=0.4,
        conf_thres=0.5,
        iou_thres=0.5,
        prop_flag=False,
        max_mask_area=200,
        dup_iou_thres=0.15,
        confirm_iou_thres=0.7,
        first_stage_thres=0.7,
        feat_size=[4,10],
        use_kalman=True,
        asso_with_motion=False,
        motion_lambda=1, 
        motion_gated=False,
    ),
    frame_rate=5, # for pvsg
)
