_base_ = [
    '../_base_/models/resnet50_fmow.py', '../_base_/datasets/fmow_bs64.py',
    '../_base_/schedules/aid_bs64_SGD.py', '../_base_/default_runtime.py'
]
evaluation = dict(
    interval=10, metric='accuracy',
    save_best='auto')  # save the checkpoint with highest accuracy

expr_name = 'resnet50_b64_100e_fmow'
init_kwargs = dict(
    project='rsi_cls',
    entity='tum-tanmlh',
    name=expr_name,
    resume='never'
)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
        dict(type='MMClsWandbHook',
             init_kwargs=init_kwargs,
             interval=201,
             num_eval_images=20),
        # dict(type='PseudoLabelingHook',
        #      log_dir='work_dirs/pseudo_labels/deeplabv3plus_r50-d8_512x512_80k_loveda_r2u',
        #      interval=1),
    ])

