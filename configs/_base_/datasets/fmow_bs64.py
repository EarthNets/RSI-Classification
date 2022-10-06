# dataset settings
dataset_type = 'EODataset'
datapipe = 'fmow'
data_root = '../../Datasets/Dataset4EO/fMoW'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (224, 224)
#crop_size = (512, 512)
train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='RandomResizedCrop', size=crop_size),
        # dict(type='Resize', size=(224, 224), ratio_range=(0.5, 2.0)),
        # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
        dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
        # dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **img_norm_cfg),
        # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
        # dict(type='DefaultFormatBundle'),
        # dict(type='Collect', keys=['img', 'gt_semantic_seg']),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='ToTensor', keys=['gt_label']),
        dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', size=crop_size),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img'])
]
data = dict(
        samples_per_gpu=64,
        workers_per_gpu=16,
        train=dict(
                    type=dataset_type,
                    data_root=data_root,
                    datapipe=datapipe,
                    split='train',
                    pipeline=train_pipeline),
        val=dict(
                    type=dataset_type,
                    data_root=data_root,
                    datapipe=datapipe,
                    split='val',
                    pipeline=test_pipeline),
        test=dict(
                    type=dataset_type,
                    data_root=data_root,
                    datapipe=datapipe,
                    split='val',
                    pipeline=test_pipeline),
        )

