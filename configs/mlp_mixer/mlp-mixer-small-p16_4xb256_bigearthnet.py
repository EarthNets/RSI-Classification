_base_ = [
    #'../_base_/models/mlp_mixer_base_patch16.py',
    '../_base_/datasets/bigearthnet_bs256_rgb.py',
    #'../_base_/schedules/imagenet_bs4096_AdamW.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MlpMixer',
        arch='s',
        img_size=224,
        patch_size=16,
        drop_rate=0.1,
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]),
    neck=dict(type='GlobalAveragePooling', dim=1),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=19,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
)

# specific to vit pretrain
paramwise_cfg = dict(custom_keys={
    '.cls_token': dict(decay_mult=0.0),
    '.pos_embed': dict(decay_mult=0.0)
})

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.003,
    weight_decay=0.3,
    paramwise_cfg=paramwise_cfg,
)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=10,
    #warmup_iters=3000,
    warmup_ratio=1e-4,
)
runner = dict(type='EpochBasedRunner', max_epochs=100)