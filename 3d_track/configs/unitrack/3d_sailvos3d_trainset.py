_base_ = [
    '../mask2former/mask2former_r50_lsj_8x2_50e_coco-panoptic_custom_singel_video_test.py'
]

tracker_cfg = dict(
    test_list_file = '/mnt/lustre/share/jkyang/psg4d/dataset_jcenaa/dataset/sailvos3d_2/train_list.txt',
    save_prefix = '/mnt/lustre/jkyang/PSG4D/share/dataset_jcenaa/test',
    segmentation_path = '/mnt/lustre/share/jcen/semantic_track_sailvos3d',
    dataset = 'sailvos3d',
    common=dict(
        exp_name="imagenet_resnet50_s3_womotion_timecycle",
        model_type="imagenet50",
        remove_layers=['layer4'],
        im_mean=[0.485, 0.456, 0.406],
        im_std=[0.229, 0.224, 0.225],
        nopadding=False,
        resume='/mnt/lustre/jkyang/wxpeng/CVPR23/UniTrack/pretrained/timecycle.pth',
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
        use_kalman=False,
        asso_with_motion=False,
        motion_lambda=1, 
        motion_gated=False,
    ),
    frame_rate=5, # for pvsg
)
