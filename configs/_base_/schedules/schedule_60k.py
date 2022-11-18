# optimizer
optimizer = dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.00005)
# optimizer = dict(type='SGD', lr=5e-7, momentum=0.9, weight_decay=0.00005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=5e-7, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=60000)
checkpoint_config = dict(by_epoch=False, interval=6000)
evaluation = dict(interval=5999, metric='mIoU')
