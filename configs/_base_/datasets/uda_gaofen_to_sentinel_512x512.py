# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# dataset settings
dataset_type = 'SentinelDataset'
data_root = 'data/sentinel/'
img_norm_cfg = dict(
    # mean=[135.37006826,114.89351608,123.00871974], std=[15.94130083,17.44040382,22.59554214], to_rgb=True)  #这里要换上我们数据集的参数
        mean=[0,0,0], std=[255,255,255], to_rgb=True) 

crop_size = (320, 320)
gaofen_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1280, 720), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),

    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
sentinel_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 512),ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 320),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='gaofenDataset',
            data_root='data/gaofen/',
            img_dir='image',
            ann_dir='label',
            pipeline=gaofen_train_pipeline),
        target=dict(
            type='sentinelDataset',
            data_root='data/sentinel/',
            img_dir='image',
            ann_dir='label',
            split="train.txt",
            pipeline=sentinel_train_pipeline)),
    val=dict(                                    #需要把目标域设置为sentinel影像
        type='sentinelDataset',
        data_root='data/sentinel/',
        img_dir='image',
        ann_dir='label',
        split="val.txt",
        pipeline=test_pipeline),
    test=dict(
        type='sentinelDataset',
        data_root='data/sentinel/',
        img_dir='image',
        ann_dir='label',
        split="test.txt",
        pipeline=test_pipeline))
