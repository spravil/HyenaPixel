from functools import partial

import timm
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model

from hyenapixel.metaformer import Attention, MetaFormer, MlpHead, SepConv
from hyenapixel.nn import HyenaOperator, HyenaPixelOperator


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": (7, 7),
        "crop_pct": 1.0,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "hpx_former_s18": _cfg(
        url="https://huggingface.co/Spravil/hpx_former_s18.westai_in1k/resolve/main/pytorch_model.bin",
        tag="westai_in1k",
    ),
    "hpx_former_s18_384": _cfg(
        url="https://huggingface.co/Spravil/hpx_former_s18.westai_in1k_ema_384/resolve/main/pytorch_model.bin",
        tag="westai_in1k_384",
        input_size=(3, 384, 384),
    ),
    "hpx_former_b36": _cfg(
        url="https://huggingface.co/Spravil/hpx_former_b36.westai_in1k/resolve/main/pytorch_model.bin",
        tag="westai_in1k",
    ),
    "hpx_a_former_s18": _cfg(
        url="https://huggingface.co/Spravil/hpx_a_former_s18.westai_in1k/resolve/main/pytorch_model.bin",
        tag="westai_in1k",
    ),
    "hb_former_s18": _cfg(
        url="https://huggingface.co/Spravil/hb_former_s18.westai_in1k/resolve/main/pytorch_model.bin", tag="westai_in1k"
    ),
    "hb_former_b36": _cfg(
        url="https://huggingface.co/Spravil/hb_former_b36.westai_in1k/resolve/main/pytorch_model.bin",
        tag="westai_in1k",
    ),
    "c_hpx_former_s18": _cfg(
        url="https://huggingface.co/Spravil/c_hpx_former_s18.westai_in1k/resolve/main/pytorch_model.bin",
        tag="westai_in1k",
    ),
    "hb_a_former_s18": _cfg(
        url="https://huggingface.co/Spravil/hb_a_former_s18.westai_in1k/resolve/main/pytorch_model.bin",
        tag="westai_in1k",
    ),
}


def _load_checkpoint(model, model_name, num_features, prepare_state_dict_fn=None, strict=True):
    model.pretrained_cfg = default_cfgs[model_name]
    model.pretrained_cfg["architecture"] = model_name
    model.num_features = num_features
    state_dict = torch.hub.load_state_dict_from_url(
        url=model.pretrained_cfg["url"],
        file_name=f"{model_name}_{model.pretrained_cfg['tag']}.pth",
        map_location="cpu",
        check_hash=True,
    )
    if prepare_state_dict_fn is not None:
        state_dict = prepare_state_dict_fn(model, state_dict)
    model.load_state_dict(state_dict, strict=strict)


def _create_metaformer(
    name,
    depths,
    dims,
    token_mixers=None,
    pretrained=False,
    token_mixers_args=None,
    strict=True,
    prepare_state_dict_fn=None,
    model_prefix="",
    **kwargs,
):
    if token_mixers is None:
        if token_mixers_args is None:
            token_mixers_args = dict()
        token_mixers = [
            partial(HyenaPixelOperator, filter_emb_dim=32, long_kernel_size=111, **token_mixers_args),
            partial(HyenaPixelOperator, filter_emb_dim=32, long_kernel_size=55, **token_mixers_args),
            partial(HyenaPixelOperator, filter_emb_dim=48, long_kernel_size=27, **token_mixers_args),
            partial(HyenaPixelOperator, filter_emb_dim=64, long_kernel_size=13, **token_mixers_args),
        ]
    model = MetaFormer(depths=depths, dims=dims, token_mixers=token_mixers, head_fn=MlpHead, **kwargs)
    if pretrained:
        _load_checkpoint(
            model,
            name + model_prefix,
            num_features=dims[-1],
            prepare_state_dict_fn=prepare_state_dict_fn,
            strict=strict,
        )
    return model


@register_model
def hpx_former_s18(pretrained=False, **kwargs):
    token_mixers_args = dict(use_layernorm=True)
    model = _create_metaformer(
        name="hpx_former_s18",
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        pretrained=pretrained,
        token_mixers_args=token_mixers_args,
        **kwargs,
    )
    return model


@register_model
def hpx_former_s18_384(pretrained=False, **kwargs):
    token_mixers_args = dict(use_layernorm=True)
    model = _create_metaformer(
        name="hpx_former_s18_384",
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers_args=token_mixers_args,
        pretrained=pretrained,
        **kwargs,
    )
    return model


def interpolate_hpx_former_s18_state_dict(model, state_dict):
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model_state_dict = model.state_dict()
    new_state_dict = dict()
    for k in state_dict:
        if k.endswith("long_conv.z") and model_state_dict[k].shape != state_dict[k].shape:
            print(f"Reshape {k}: {state_dict[k].shape} to {model_state_dict[k].shape}")
            x = state_dict[k]
            orig_dtype = x.dtype
            x = x.float()
            x = F.interpolate(x, size=model_state_dict[k].shape[2:], mode="bicubic", antialias=True)
            x = x.to(orig_dtype)
            state_dict[k] = x

        if k.endswith("long_conv.distance_map") and model_state_dict[k].shape != state_dict[k].shape:
            from hyenapixel.nn.hyenapixel import create_distance_tensor

            print(f"Reshape {k}: {state_dict[k].shape} to {model_state_dict[k].shape}")
            old_shape = torch.tensor(state_dict[k].shape)[-1]
            state_dict[k] = create_distance_tensor(model_state_dict[k].shape[2:])[None, None]
            new_shape = torch.tensor(state_dict[k].shape)[-1] / 2
            state_dict[k] /= new_shape / old_shape

        new_state_dict[k] = state_dict[k]
    return new_state_dict


@register_model
def hpx_former_s18_384_interpolate(pretrained=False, **kwargs):
    token_mixers = [
        partial(HyenaPixelOperator, filter_emb_dim=32, long_kernel_size=191, use_layernorm=True),
        partial(HyenaPixelOperator, filter_emb_dim=32, long_kernel_size=95, use_layernorm=True),
        partial(HyenaPixelOperator, filter_emb_dim=48, long_kernel_size=47, use_layernorm=True),
        partial(HyenaPixelOperator, filter_emb_dim=64, long_kernel_size=23, use_layernorm=True),
    ]
    model = _create_metaformer(
        name="hpx_former_s18",
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=token_mixers,
        pretrained=pretrained,
        prepare_state_dict_fn=interpolate_hpx_former_s18_state_dict,
        **kwargs,
    )
    return model


@register_model
def hpx_former_b36(pretrained=False, **kwargs):
    token_mixers_args = dict(use_layernorm=True)
    model = _create_metaformer(
        name="hpx_former_b36",
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        pretrained=pretrained,
        token_mixers_args=token_mixers_args,
        **kwargs,
    )
    return model


@register_model
def hpx_a_former_s18(pretrained=False, **kwargs):
    token_mixers = [
        partial(HyenaPixelOperator, filter_emb_dim=32, long_kernel_size=111),
        partial(HyenaPixelOperator, filter_emb_dim=32, long_kernel_size=55),
        Attention,
        Attention,
    ]
    model = _create_metaformer(
        name="hpx_a_former_s18",
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=token_mixers,
        pretrained=pretrained,
        **kwargs,
    )
    return model


@register_model
def c_hpx_former_s18(pretrained=False, **kwargs):
    token_mixers = [
        SepConv,
        SepConv,
        partial(HyenaPixelOperator, filter_emb_dim=48, long_kernel_size=27),
        partial(HyenaPixelOperator, filter_emb_dim=64, long_kernel_size=13),
    ]
    model = _create_metaformer(
        name="c_hpx_former_s18",
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=token_mixers,
        pretrained=pretrained,
        **kwargs,
    )
    return model


@register_model
def hb_former_s18(pretrained=False, **kwargs):
    token_mixers = [
        partial(HyenaOperator, emb_dim=33, filter_order=64, l_max=3136, is_causal=False),
        partial(HyenaOperator, emb_dim=33, filter_order=64, l_max=784, is_causal=False),
        partial(HyenaOperator, emb_dim=49, filter_order=96, l_max=196, is_causal=False),
        partial(HyenaOperator, emb_dim=65, filter_order=128, l_max=49, is_causal=False),
    ]
    model = _create_metaformer(
        name="hb_former_s18",
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=token_mixers,
        pretrained=pretrained,
        **kwargs,
    )
    return model


@register_model
def hb_former_b36(pretrained=False, **kwargs):
    token_mixers = [
        partial(HyenaOperator, emb_dim=33, filter_order=64, l_max=3136, is_causal=False, use_layernorm=True),
        partial(HyenaOperator, emb_dim=33, filter_order=64, l_max=784, is_causal=False, use_layernorm=True),
        partial(HyenaOperator, emb_dim=49, filter_order=96, l_max=196, is_causal=False, use_layernorm=True),
        partial(HyenaOperator, emb_dim=65, filter_order=128, l_max=49, is_causal=False, use_layernorm=True),
    ]
    model = _create_metaformer(
        name="hb_former_b36",
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=token_mixers,
        pretrained=pretrained,
        **kwargs,
    )
    return model


@register_model
def hb_a_former_s18(pretrained=False, **kwargs):
    token_mixers = [
        partial(HyenaOperator, emb_dim=33, filter_order=64, l_max=3136, is_causal=False),
        partial(HyenaOperator, emb_dim=33, filter_order=64, l_max=784, is_causal=False),
        Attention,
        Attention,
    ]
    model = _create_metaformer(
        name="hb_a_former_s18",
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=token_mixers,
        pretrained=pretrained,
        **kwargs,
    )
    return model


@register_model
def combi_former_s18_2(pretrained=False, **kwargs):
    class CombiModel(nn.Module):
        def __init__(self, pretrained) -> None:
            super().__init__()
            self.num_classes = 1000
            self.m1 = timm.create_model("convformer_s18", pretrained=pretrained)
            self.m2 = timm.create_model("hpx_former_s18", pretrained=pretrained)

        def forward(self, x):
            return torch.mean(torch.stack([self.m1(x), self.m2(x)]), dim=0)

    return CombiModel(pretrained)


@register_model
def combi_former_s18_4(pretrained=False, **kwargs):
    class CombiModel(nn.Module):
        def __init__(self, pretrained) -> None:
            super().__init__()
            self.num_classes = 1000
            self.m1 = timm.create_model("convformer_s18", pretrained=pretrained)
            self.m2 = timm.create_model("hpx_former_s18", pretrained=pretrained)
            self.m3 = timm.create_model("hb_former_s18", pretrained=pretrained)
            self.m4 = timm.create_model("caformer_s18", pretrained=pretrained)

        def forward(self, x):
            return torch.mean(torch.stack([self.m1(x), self.m2(x), self.m3(x), self.m4(x)]), dim=0)

    return CombiModel(pretrained)
