_base_ = [
    '../_base_/models/convnext/convnext-tiny.py',
    '../_base_/datasets/bigearthnet_bs256_swin_224.py',
    '../_base_/schedules/bigearthnet_bs1024_adamw_swin.py',
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
        type='MultiLabelLinearClsHead',
        num_classes=19,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, use_sigmoid=True)
    ))

data = dict(samples_per_gpu=256)

optimizer = dict(lr=4e-3)

custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]
