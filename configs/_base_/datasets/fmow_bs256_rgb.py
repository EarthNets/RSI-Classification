# dataset settings
dataset_type = 'FMoW'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
#img_norm_cfg = dict(
#    mean=[0, 0, 0],
#    std=[255, 255, 255],
#    to_rgb=True)    
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    #dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=256),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=10,
    train=dict(
        type=dataset_type, 
        data_prefix='/p/project/hai_ssl4eo/wang_yi/data/fmow/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='/p/project/hai_ssl4eo/wang_yi/data/fmow/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix='/p/scratch/hai_ssl4eo/data/fmow/test_new',
        pipeline=test_pipeline))
evaluation = dict(
    interval=5, metric='accuracy', save_best='auto')