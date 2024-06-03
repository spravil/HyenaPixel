_base_ = [
    "../_base_/models/upernet_r50.py",
    "../_base_/datasets/ade20k.py",
    "../_base_/schedules/schedule_160k.py",
    "../_base_/default_runtime.py",
]

custom_imports = dict(imports=["mmseg.models.backbones.timm"], allow_failed_imports=False)


crop_size = (512, 512)

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
    decode_head=dict(in_channels=[64, 128, 320, 512], num_classes=150),
    auxiliary_head=dict(in_channels=320, num_classes=150),
    data_preprocessor=dict(size=crop_size),
    pretrained=None,
)

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type="AmpOptimWrapper",
    constructor="LearningRateDecayOptimizerConstructor",
    paramwise_cfg={"decay_rate": 0.7, "decay_type": "layer_wise", "num_layers": 6},
    optimizer=dict(type="AdamW", lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    loss_scale="dynamic",
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type="PolyLR", power=1.0, begin=1500, end=160000, eta_min=0.0, by_epoch=False),
]

# By default, models are trained on 4 GPUs with 4 images per GPU
train_dataloader = dict(batch_size=4, num_workers=8)
