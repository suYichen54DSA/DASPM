# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# dataset settings
# dataset_type = 'LoveDADataset'
# data_root = ''
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)
crop_size = (512, 512)
loveda_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),    
    # dict(type='Resize', img_scale=(50, 50)),
    # dict(type='Resize', img_scale=(256, 256)),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
whdld_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='Resize', img_scale=(50, 50)),
    # dict(type='Resize', img_scale=(256, 256)),
    dict(type='Resize', img_scale=(1280, 720), ratio_range=(0.5, 2.0)),
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
    # dict(type='Resize', img_scale=(50, 50)),
    # dict(type='Resize', img_scale=(256, 256)),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
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
    samples_per_gpu=6,
    workers_per_gpu=8,
    train=dict(
        type='UDADataset',
        source=dict(
            type='LoveDADataset',
            data_root='/home/heda/xjq/Research/Data/Segmentation/LoveDA/',
            img_dir='image_spm/Urban',
            ann_dir='label_convert',
            # split="temp.txt",
            pipeline=loveda_train_pipeline),
        target=dict(
            type='WHDLDataset',
            data_root='/home/heda/xjq/Research/Data/Segmentation/WHDLD/',
            img_dir='image_spm',
            ann_dir='label_loveda',
            # split="train.txt",
            pipeline=whdld_train_pipeline)),
    val=dict(
        type='WHDLDataset',
        data_root='/home/heda/xjq/Research/Data/Segmentation/WHDLD/',
        img_dir='image_spm',
        ann_dir='label_loveda',
        split="val_da.txt",
        pipeline=test_pipeline),
    test=dict(
        type='WHDLDataset',
        data_root='/home/heda/xjq/Research/Data/Segmentation/WHDLD/',
        img_dir='image_spm',
        ann_dir='label_loveda',
        split="test_da.txt",
        pipeline=test_pipeline))
