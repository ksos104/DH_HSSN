_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8_vd_contrast.py', '../_base_/datasets/pascal_person_part.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_60k.py'
]
model = dict(pretrained='/mnt/server14_hard0/msson/HSSN_pytorch/checkpoints/deeplabv3plus_r101-d8_480x480_60k_pascal_person_part_hiera_triplet.pth', backbone=dict(depth=101),
             decode_head=dict(num_classes=12,loss_decode=dict(type='RMIHieraTripletLoss',num_classes=7, loss_weight=1.0)),
             auxiliary_head=dict(num_classes=7),
             test_cfg=dict(mode='whole', is_hiera=True, hiera_num_classes=5))
