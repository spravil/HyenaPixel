import torch
from einops import pack, rearrange, repeat
from timm.layers import AttentionPoolLatent
from timm.models.metaformer import SquaredReLU
from torch import nn


class AttentionPoolHead(nn.Module):
    """MLP classification head"""

    def __init__(
        self,
        dim,
        num_classes=1000,
        mlp_ratio=4,
        act_layer=SquaredReLU,
        norm_layer=nn.LayerNorm,
        head_dropout=0.0,
        bias=True,
        use_register_tokens=False,
        num_register_tokens=4,
    ):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn_pool = AttentionPoolLatent(dim, num_heads=dim // 32, mlp_ratio=mlp_ratio, norm_layer=norm_layer)
        self.fc_norm = norm_layer(dim)
        self.fc = nn.Linear(dim, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)
        self.use_register_tokens = use_register_tokens
        if self.use_register_tokens:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))

    def forward(self, x):
        x = rearrange(x, "b h w c -> b (h w) c")
        if self.use_register_tokens:
            r = repeat(self.register_tokens, "n d -> b n d", b=x.shape[0])
            x, _ = pack([x, r], "b * c")  # * marks the dimension where the packing happens
        x = self.attn_pool(self.norm(x))
        x = self.fc_norm(x)
        x = self.head_dropout(x)
        x = self.fc(x)
        return x
