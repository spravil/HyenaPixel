import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from hyenapixel.metaformer import LayerNormWithoutBias
from hyenapixel.nn.functional import fftconv2d


class Sin2D(nn.Module):
    def __init__(self, dim, w=10):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return torch.sin(self.freq[:, :] * x)


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with " "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1 :: 2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


def positionalencoding2d_from_1d(d_model, height, width):
    seq_len = height * width
    # The time embedding fed to the filteres is normalized so that t_f = 1
    t = torch.linspace(0, 1, seq_len // 2 + 1)
    t = torch.cat([t[:-1], t.flip(dims=(0,))])[None, :, None]
    if d_model > 1:
        bands = (d_model - 1) // 2
    # To compute the right embeddings we use the "proper" linspace
    t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
    w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1
    f = torch.linspace(1e-4, bands - 1, bands)[None, None]
    z = torch.exp(-1j * f * w)
    z = torch.cat([t, z.real, z.imag], dim=-1)
    z = rearrange(z[0], "(h w) c -> c h w", h=height, w=width)
    return z


def create_distance_tensor(size):
    rows, cols = size
    tensor = torch.zeros((rows, cols))
    center_row = rows / 2
    center_col = cols / 2
    for i in range(rows):
        for j in range(cols):
            distance = ((i + 0.5 - center_row) ** 2 + (j + 0.5 - center_col) ** 2) ** 0.5
            tensor[i, j] = distance
    return tensor


class Hyena2DConv(nn.Module):
    def __init__(self, dim, kernel_size=55, filter_emb_dim=32, filter_order=64, use_decay=True, pe_type="2d"):
        super().__init__()
        self.use_decay = use_decay

        pe = None
        if pe_type == "2d":
            pe = positionalencoding2d
        elif pe_type == "1d":
            pe = positionalencoding2d_from_1d
            filter_emb_dim += 1
        else:
            raise NotImplementedError(f"Positional encoding of type {pe_type} does not exist")
        self.z = nn.Parameter(pe(filter_emb_dim, kernel_size, kernel_size).unsqueeze(0))

        if self.use_decay:
            fast_decay_pct = 0.3
            slow_decay_pct = 1.5
            target = 1e-2
            max_decay = math.log(target) / fast_decay_pct
            min_decay = math.log(target) / slow_decay_pct
            deltas = torch.linspace(min_decay, max_decay, dim)[None, :, None, None]
            self.distance_map = nn.Parameter(
                create_distance_tensor((kernel_size, kernel_size))[None, None], requires_grad=False
            )
            self.deltas = nn.Parameter(deltas / kernel_size)

            self.shift = nn.Parameter(torch.zeros(dim)[None, :, None, None])

        # Learnable bias
        self.bias = nn.Parameter(torch.randn(dim)[None, :, None, None])

        # Create implicit filter function
        self.implicit_filter = nn.Sequential(
            nn.Conv2d(filter_emb_dim, filter_order, kernel_size=1),
            Sin2D(dim=filter_order, w=1),
            nn.Conv2d(filter_order, filter_order, kernel_size=1),
            Sin2D(dim=filter_order, w=1),
            nn.Conv2d(filter_order, dim, kernel_size=1, bias=False),
        )

    def forward(self, x):
        k = self.implicit_filter(self.z)

        if self.use_decay:
            decay = torch.exp(-self.distance_map * self.deltas.abs())
            k = k * (decay + self.shift)

        return fftconv2d(x, k, self.bias)


class HyenaPixelOperator(nn.Module):
    def __init__(
        self,
        dim=768,
        drop=0.0,
        kernel_size=5,
        use_seperable_conv=False,
        long_kernel_size=111,
        use_implicit_filter=True,
        sparsity=0,
        filter_emb_dim=32,
        use_decay=True,
        filter_order=None,
        use_layernorm=False,
        pe_type="2d",
    ):
        super().__init__()
        self.use_layernorm = use_layernorm
        if filter_order is None:
            filter_order = 2 * filter_emb_dim
        self.dim = dim
        self.pointwise_conv = nn.Conv2d(in_channels=dim, out_channels=3 * dim, kernel_size=1)
        self.depthwise_conv = nn.Conv2d(
            in_channels=3 * dim, out_channels=3 * dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=3 * dim
        )
        self.long_kernel_size = long_kernel_size
        self.use_seperable_conv = use_seperable_conv
        self.use_implicit_filter = use_implicit_filter
        self.sparsity = sparsity
        self.dropconnect = nn.Dropout(self.sparsity) if sparsity > 0 else None
        if not use_implicit_filter:
            if not use_seperable_conv:
                self.long_conv = nn.Conv2d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=(long_kernel_size, long_kernel_size),
                    padding=(long_kernel_size // 2, long_kernel_size // 2),
                    groups=dim,
                )
            else:
                self.long_horizontal_conv = nn.Conv2d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=(long_kernel_size, 1),
                    padding=(long_kernel_size // 2, 0),
                    groups=dim,
                )
                self.long_vertical_conv = nn.Conv2d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=(1, long_kernel_size),
                    padding=(0, long_kernel_size // 2),
                    groups=dim,
                )
        else:
            if not use_seperable_conv:
                self.long_conv = Hyena2DConv(
                    dim,
                    kernel_size=long_kernel_size,
                    filter_emb_dim=filter_emb_dim,
                    filter_order=filter_order,
                    use_decay=use_decay,
                    pe_type=pe_type,
                )
            else:
                raise NotImplementedError()

        if use_layernorm:
            self.qk_norm = LayerNormWithoutBias(dim)
            self.out_norm = LayerNormWithoutBias(dim)
            self.out_proj = nn.Linear(in_features=dim, out_features=dim)
        else:
            self.out_proj = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.depthwise_conv(self.pointwise_conv(x))
        q, k, v = x.split(self.dim, dim=1)

        qk = q * k

        if self.use_layernorm:
            qk = qk.permute(0, 2, 3, 1)
            qk = self.qk_norm(qk)
            qk = qk.permute(0, 3, 1, 2)

        if not self.use_seperable_conv:
            if self.sparsity > 0 and not self.use_implicit_filter:
                weight = self.dropconnect(self.long_conv.weight)
                x = v * F.conv2d(
                    qk,
                    weight,
                    self.long_conv.bias,
                    self.long_conv.stride,
                    self.long_conv.padding,
                    self.long_conv.dilation,
                    self.long_conv.groups,
                )

            else:
                x = v * self.long_conv(qk)
        else:
            if self.sparsity > 0:
                weighth = self.dropconnect(self.long_horizontal_conv.weight)
                weightv = self.dropconnect(self.long_vertical_conv.weight)
                qk = F.conv2d(
                    qk,
                    weighth,
                    self.long_horizontal_conv.bias,
                    self.long_horizontal_conv.stride,
                    self.long_horizontal_conv.padding,
                    self.long_horizontal_conv.dilation,
                    self.long_horizontal_conv.groups,
                )
                x = v * F.conv2d(
                    qk,
                    weightv,
                    self.long_vertical_conv.bias,
                    self.long_vertical_conv.stride,
                    self.long_vertical_conv.padding,
                    self.long_vertical_conv.dilation,
                    self.long_vertical_conv.groups,
                )

            else:
                x = v * self.long_vertical_conv(self.long_horizontal_conv(qk))

        if self.use_layernorm:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = self.out_norm(x)
            x = self.out_proj(x)
        else:
            x = self.out_proj(x)
            x = x.permute(0, 2, 3, 1).contiguous()
        return x
