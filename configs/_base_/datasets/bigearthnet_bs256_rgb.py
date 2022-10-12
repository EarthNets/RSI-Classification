# dataset settings
dataset_type = 'BigEarthNet'
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
    dict(type='RandomResizedCrop', size=224, scale=(0.8,1.0)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
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
        data_prefix='/p/project/hai_ssl4eo/wang_yi/data/BigEarthNet-v1.0-RGB',
        ann_file='/p/project/hai_ssl4eo/wang_yi/data/19_labels_train.csv',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='/p/project/hai_ssl4eo/wang_yi/data/BigEarthNet-v1.0-RGB',
        ann_file='/p/project/hai_ssl4eo/wang_yi/data/19_labels_val.csv',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix='/p/project/hai_ssl4eo/wang_yi/data/BigEarthNet-v1.0-RGB',
        ann_file='/p/project/hai_ssl4eo/wang_yi/data/19_labels_test.csv',
        pipeline=test_pipeline,
        test_mode=True))
evaluation = dict(
    interval=5, metric=['mAP', 'CP', 'OP', 'CR', 'OR', 'CF1', 'OF1'])