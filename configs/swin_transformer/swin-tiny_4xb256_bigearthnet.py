_base_ = [
    #'../_base_/models/swin_transformer/tiny_224.py',
    '../_base_/datasets/bigearthnet_bs256_swin_224.py',
    #'../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer',
        arch='tiny',
        img_size=224,
        drop_path_rate=0.2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrained_weights/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth',
            prefix='backbone')        
        ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=19,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=19, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=19, prob=0.5)
    ]))
    
    
# specific to vit pretrain
paramwise_cfg = dict(custom_keys={
    '.cls_token': dict(decay_mult=0.0),
    '.pos_embed': dict(decay_mult=0.0)
})

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.001,
    weight_decay=0.3,
    paramwise_cfg=paramwise_cfg,
)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=5,
    warmup_ratio=1e-4,
)
runner = dict(type='EpochBasedRunner', max_epochs=100)