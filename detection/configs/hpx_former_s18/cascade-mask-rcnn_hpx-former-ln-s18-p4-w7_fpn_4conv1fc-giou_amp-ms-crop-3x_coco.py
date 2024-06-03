_base_ = [
    "../_base_/models/cascade-mask-rcnn_r50_fpn.py",
    "../_base_/datasets/coco_instance.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]

custom_imports = dict(imports=["mmdet.models.backbones.timm"], allow_failed_imports=False)

model = dict(
    backbone=dict(
        _delete_=True,
        type="TimmModel",
        model_name="hpx_former_s18",
        drop_path_rate=0.4,
        out_indices=[0, 1, 2, 3],
        features_only=True,
        pretrained=True,
        strict=False,
    ),
    neck=dict(in_channels=[64, 128, 320, 512]),
    roi_head=dict(
        bbox_head=[
            dict(
                type="ConvFCBBoxHead",
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[0.1, 0.1, 0.2, 0.2]
                ),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type="SyncBN", requires_grad=True),
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type="GIoULoss", loss_weight=10.0),
            ),
            dict(
                type="ConvFCBBoxHead",
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[0.05, 0.05, 0.1, 0.1]
                ),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type="SyncBN", requires_grad=True),
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type="GIoULoss", loss_weight=10.0),
            ),
            dict(
                type="ConvFCBBoxHead",
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067],
                ),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type="SyncBN", requires_grad=True),
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type="GIoULoss", loss_weight=10.0),
            ),
        ]
    ),
)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args={{_base_.backend_args}}),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="RandomChoice",
        transforms=[
            [
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    keep_ratio=True,
                )
            ],
            [
                dict(type="RandomChoiceResize", scales=[(400, 1333), (500, 1333), (600, 1333)], keep_ratio=True),
                dict(type="RandomCrop", crop_type="absolute_range", crop_size=(384, 600), allow_negative_crop=True),
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    keep_ratio=True,
                ),
            ],
        ],
    ),
    dict(type="PackDetInputs"),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

max_epochs = 36
train_cfg = dict(max_epochs=max_epochs)

# learning rate
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type="MultiStepLR", begin=0, end=max_epochs, by_epoch=True, milestones=[27, 33], gamma=0.1),
]

# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper = dict(
    type="AmpOptimWrapper",
    constructor="LearningRateDecayOptimizerConstructor",
    paramwise_cfg={"decay_rate": 0.7, "decay_type": "layer_wise", "num_layers": 6},
    optimizer=dict(_delete_=True, type="AdamW", lr=0.0002, betas=(0.9, 0.999), weight_decay=0.05),
    #    clip_grad=dict(max_norm=0.01, norm_type=2),
    #    loss_scale="dynamic",
)
