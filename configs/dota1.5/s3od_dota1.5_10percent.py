_base_ = ["../../thirdparty/mmrotate/configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_for_s3od_teacher_dota15.py",
          "../../thirdparty/mmrotate/configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_for_s3od_student_dota15.py"]


model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style="caffe",
        init_cfg=dict(
            type="Pretrained", checkpoint="open-mmlab://detectron2/resnet50_caffe"
        ),
    )
)

model_s = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style="caffe",
        init_cfg=dict(
            type="Pretrained", checkpoint="open-mmlab://detectron2/resnet50_caffe"
        ),
    )
)

img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(1333, 400), (1333, 1200)],
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
            dict(
                type="OneOf",
                transforms=[
                    dict(type=k)
                    for k in [
                        "Identity",
                        "AutoContrast",
                        "RandEqualize",
                        "RandSolarize",
                        "RandColor",
                        "RandContrast",
                        "RandBrightness",
                        "RandSharpness",
                        "RandPosterize",
                    ]
                ],
            ),
        ],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="sup"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
        ),
    ),
]

strong_pipeline = [
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(1333, 400), (1333, 1200)],
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
            dict(
                type='ShuffledSequential',
                transforms=[
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='Identity'),
                            dict(type='AutoContrast'),
                            dict(type='RandEqualize'),
                            dict(type='RandSolarize'),
                            dict(type='RandColor'),
                            dict(type='RandContrast'),
                            dict(type='RandBrightness'),
                            dict(type='RandSharpness'),
                            dict(type='RandPosterize')
                        ]),
                    dict(
                        type='OneOf',
                        transforms=[{
                            'type': 'RandTranslate',
                            'x': (-0.1, 0.1)
                        }, {
                            'type': 'RandTranslate',
                            'y': (-0.1, 0.1)
                        }, 
                        {
                            'type': 'RandRotate',
                            'angle': (-30, 30)
                        },
                                    [{
                                        'type':
                                        'RandShear',
                                        'x': (-30, 30)
                                    }, {
                                        'type':
                                        'RandShear',
                                        'y': (-30, 30)
                                    }
                                    ]
                                    ])
                ]),
            # dict(
            #     type='RandErase',
            #     n_iterations=(1, 10),
            #     size=[0, 0.05],
            #     squared=True)
        ],
        record=True),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="unsup_student"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
            "transform_matrix",
        ),
    ),
]
weak_pipeline = [
     dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(1333, 400), (1333, 1200)],
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
        ],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="unsup_teacher"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
            "transform_matrix",
        ),
    ),
]
unsup_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type="LoadAnnotations", with_bbox=True),
    # generate fake labels for data format compatibility
    dict(type="PseudoSamples", with_bbox=True),
    dict(
        type="MultiBranch", unsup_student=strong_pipeline, unsup_teacher=weak_pipeline
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=5,
    workers_per_gpu=5,
    train=dict(
        _delete_=True,
        type="SemiDataset",
        sup=dict(
            type="DOTAV15Dataset",
            ann_file="data/dota1.5/coco/semi_train/semi-${fold}@${percent}/labels/",
            img_prefix="data/dota1.5/train_obb/split_images/images/",
            pipeline=train_pipeline,
        ),
        unsup=dict(
            type="DOTAV15Dataset",
            ann_file='data/dota1.5/coco/semi_train/semi-${fold}@${percent}-unlabeled/labels/',
            img_prefix='data/dota1.5/train_obb/split_images/images/',
            pipeline=unsup_pipeline,
            filter_empty_gt=False,
        ),
    ),
    val=dict(
        ann_file='data/dota1.5/val_obb/split_images/annfiles/',
        img_prefix='data/dota1.5/val_obb/split_images/images/',
        pipeline=test_pipeline
        ),
    test=dict(
        ann_file='data/dota1.5/val_obb/split_images/annfiles/',
        img_prefix='data/dota1.5/val_obb/split_images/images/',
        pipeline=test_pipeline),
    sampler=dict(
        train=dict(
            type="SemiBalanceSampler",
            sample_ratio=[1, 4],
            by_prob=True,
            # at_least_one=True,
            epoch_length=7330,
        )
    ),
)

semi_wrapper = dict(
    type="S3OD",
    teacher_model="${model}",
    student_model="${model_s}",
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.001,
        cls_thr_vaild_len = 100,
        cls_thr_percent = 'gmm',
        cls_thr_min = 0.6,
        cls_thr_max = 0.92,
        rpn_pseudo_threshold=0.9,
        cls_pseudo_threshold=0.9,
        cls_neg_gt_threshold=0.5,
        with_SAT =True,
        with_FNS =True,
        with_SRA = True,
        with_load = False,
        min_pseduo_box_size=0,
        unsup_weight=4.0,
        small_size=1024,
    ),
    test_cfg=dict(inference_on="teacher"),
)

# load_from = '/home/zrx/ssod/SoftTeacher/work_dirs/rs3od_rofaster_bs5x0.2_lr0.001_SRA_load/1/1_anchor_size_reweight(1030)/iter_44000.pth'
# load_from = '/home/zrx/ssod/SoftTeacher/work_unused/work_dirs_400_1200/baseline_rofaster_bs4_lr0.005/1/1/iter_18000_expand.pth'
# load_from = '/home/zrx/ssod/SoftTeacher/work_dirs/baseline_rofaster_bs4_lr0.005/5/1/iter_18000.pth'
custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="WeightSummary"),
    dict(type="MeanTeacher", momentum=0.999, interval=1, warm_up=0),
]
evaluation = dict(type="SubModulesDistEvalHook", interval=100*50, save_best='mAP')
optimizer = dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[1000*140, 1000*150])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=(160000))
checkpoint_config = dict(by_epoch=False, interval=100*50, max_keep_ckpts=2)

fp16 = dict(loss_scale="dynamic")
# fp16 = None
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
    ],
)

fold = 1
percent = 1

work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)

# fold = 1
# percent = 1
