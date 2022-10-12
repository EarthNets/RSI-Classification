_base_ = [
    '../_base_/models/convnext/convnext-tiny.py',
    '../_base_/datasets/fmow_bs256_swin_224.py',
    #'../_base_/schedules/bigearthnet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

model = dict(
    backbone=dict(
        #style='pytorch',
        #norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=[
            dict(
                type='TruncNormal',
                layer=['Conv2d', 'Linear'],
                std=.02,
                bias=0.),
            dict(type='Constant', layer=['LayerNorm'], val=1., bias=0.),
            dict(
                type='Pretrained',
                checkpoint='pretrained_weights/convnext-tiny_3rdparty_32xb128-noema_in1k_20220222-2908964a.pth',
                prefix='backbone')
        ]
    ),
    head=dict(
        type='LinearClsHead',
        num_classes=62,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))


### schedule
paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.absolute_pos_embed': dict(decay_mult=0.0),
        '.relative_position_bias_table': dict(decay_mult=0.0)
    })

# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
optimizer = dict(
    type='AdamW',
    lr=1e-3,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=paramwise_cfg)
optimizer_config = dict(grad_clip=dict(max_norm=5.0))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=5,
    warmup_by_epoch=True)

runner = dict(type='EpochBasedRunner', max_epochs=50)


custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]
