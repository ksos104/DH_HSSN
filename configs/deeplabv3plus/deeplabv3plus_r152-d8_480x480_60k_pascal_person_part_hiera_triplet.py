_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8_vd_contrast.py', '../_base_/datasets/pascal_person_part.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_60k.py'
]
# model = dict(pretrained='https://download.openmmlab.com/mmclassification/v0/imagenet/resnetv1d152_batch256_20200708-e79cb6a2.pth', backbone=dict(depth=152),
#              decode_head=dict(num_classes=12,loss_decode=dict(type='RMIHieraTripletLoss',num_classes=7, loss_weight=1.0)),
#              auxiliary_head=dict(num_classes=7),
#              test_cfg=dict(mode='whole', is_hiera=True, hiera_num_classes=5))


model = dict(pretrained='/mnt/server14_hard0/msson/HSSN_pytorch/checkpoints/resnetv1d152_batch256_20220826.pth', backbone=dict(depth=152),
             decode_head=dict(num_classes=12,loss_decode=dict(type='RMIHieraTripletLoss',num_classes=7, loss_weight=1.0)),
             auxiliary_head=dict(num_classes=7),
             test_cfg=dict(mode='whole', is_hiera=True, hiera_num_classes=5))
