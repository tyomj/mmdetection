_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
conv_cfg = dict(type='ConvWC2d')
norm_cfg = dict(
    type='MABN', eps=1e-5, momentum=0.98, B=2, real_B=16, warmup_iters=100)
model = dict(
    pretrained=None,
    backbone=dict(
        frozen_stages=-1,
        zero_init_residual=False,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg),
    neck=dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)))
# optimizer
optimizer = dict(paramwise_cfg=dict(norm_decay_mult=0))
optimizer_config = dict(_delete_=True, grad_clip=None)
# learning policy
lr_config = dict(warmup_ratio=0.1, step=[65, 71])
total_epochs = 73
