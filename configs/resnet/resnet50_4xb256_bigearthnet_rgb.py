_base_ = [
    '../_base_/datasets/bigearthnet_bs256_rgb.py',
     '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrained_weights/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )
        ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=19,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    ))
    
# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
#optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[30, 60, 90])
#lr_config = dict(policy='CosineAnnealing', min_lr=0)
runner = dict(type='EpochBasedRunner', max_epochs=100)