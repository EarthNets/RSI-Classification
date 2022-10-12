_base_ = [
    #'../_base_/models/vit-base-p16.py',
    '../_base_/datasets/bigearthnet_bs256_rgb_pil_autoaug.py',
    #'../_base_/schedules/imagenet_bs4096_AdamW.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='deit-s',
        img_size=224,
        patch_size=16,
        drop_rate=0.1,
        #norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
            #dict(
            #    type='Pretrained',
            #    checkpoint='pretrained_weights/',
            #    prefix='backbone')
        ]),
    neck=None,
    head=dict(
        type='VisionTransformerMultiLabelClsHead',
        #hidden_dim=2304,        
        num_classes=19,
        in_channels=384,
        loss=dict(
            type='CrossEntropyLoss', loss_weight=1.0, use_sigmoid=True),
        ),
    train_cfg=dict(
        augments=dict(type='BatchMixup', alpha=0.2, num_classes=19,
                      prob=1.))
    )


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
    warmup_iters=10,
    warmup_ratio=1e-4,
)
runner = dict(type='EpochBasedRunner', max_epochs=100)



