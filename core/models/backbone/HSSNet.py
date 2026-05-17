import numbers
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

from mamba_ssm import Mamba

class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"

def convert_ln_to_dyt(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicTanh(module.normalized_shape, not isinstance(module, LayerNorm2d))
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child))
    del module
    return module_output

class MambaAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.dim = dim
        self.norm = LayerNorm(dim)
        
        self.mamba = Mamba(
            d_model=dim,     
            d_state=32,      
            d_conv=8,        
            expand=4        
        )
        
        print('[MambaAttn]:', 'dims=', dim, ', num_heads=', num_heads)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        
        x = self.norm(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.mamba(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        
        return residual + x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias"):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# -------------------------------------------------------------------------
# Texture-Aware Gated Module (TAGM) - 原 MDGM
# -------------------------------------------------------------------------
class TAGM(nn.Module):
    def __init__(self, dim, hidden_features=None, dropout=0., bias=False):
        super().__init__()
        hidden_features = hidden_features or dim

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv_shallow = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                        groups=hidden_features, bias=bias)
        self.dwconv_deep1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                      groups=hidden_features, bias=bias)
        self.dwconv_deep2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                      groups=hidden_features, bias=bias)

        self.dwconv_gate = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                     groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x, x_gate = self.project_in(x).chunk(2, dim=1)
        # shallow
        x_shallow = self.dwconv_shallow(x)
        # deep
        x_deep = self.dwconv_deep2(F.gelu(self.dwconv_deep1(x)))
        x = x_shallow + x_deep
        # gated
        x_gate = F.gelu(self.dwconv_gate(x_gate))

        x = x_gate * x
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, use_dyt=False, dyt_alpha_init=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.use_dyt = use_dyt
        if self.use_dyt:
            self.dyt = DynamicTanh(normalized_shape=dim, channels_last=False, alpha_init_value=dyt_alpha_init)

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=qkv_bias)

        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

        print('[chattn]:', 'dims=', dim, ', num_heads=', num_heads, ', use_dyt=', self.use_dyt)

    def forward(self, x):
        b, c, h, w = x.shape

        if getattr(self, 'use_dyt', False):
            x = self.dyt(x)

        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        x = (attn @ v)
        x = rearrange(x, 'b head c (h w) -> b (head c) h w', h=h, w=w)

        x = self.proj(x)
        return x

# -------------------------------------------------------------------------
# Hybrid State Space Block (HSSB) - 原 TransformerBlock
# -------------------------------------------------------------------------
class HSSB(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2., qkv_bias=False, dropout=0., bias=False):
        super().__init__()
        self.ch_attn = MambaAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        
        self.norm2 = LayerNorm(dim=dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = TAGM(dim=dim, hidden_features=mlp_hidden_dim, dropout=dropout, bias=bias)

    def forward(self, x):
        x = self.ch_attn(x)
        x = x + self.ffn(self.norm2(x))
        return x

# -------------------------------------------------------------------------
# Cross-Scale Feature Fusion Module (CFFM) - 原 FeatureFusion
# -------------------------------------------------------------------------
class CFFM(nn.Module):
    def __init__(self, in_ch_high, in_ch_low, out_ch, num_heads=4, bias=False, init_gate_bias=-1.0):
        super().__init__()
        self.proj_high = nn.Conv2d(in_ch_high, out_ch, kernel_size=1, bias=bias)
        self.proj_low  = nn.Conv2d(in_ch_low,  out_ch, kernel_size=1, bias=bias)
        
        self.attn_high = Attention(out_ch, num_heads=num_heads, qkv_bias=bias, use_dyt=True)
        self.attn_low  = Attention(out_ch, num_heads=num_heads, qkv_bias=bias, use_dyt=True)
        
        self.gate_conv = nn.Conv2d(out_ch * 2, 1, kernel_size=1, bias=True)
        nn.init.constant_(self.gate_conv.bias, init_gate_bias)
        
        self.tagm = TAGM(out_ch, hidden_features=out_ch, bias=bias)
        self.mix  = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=bias)

    def forward(self, feat_high, feat_low):
        if feat_low.shape[-2:] != feat_high.shape[-2:]:
            feat_low = F.interpolate(
                feat_low, size=feat_high.shape[-2:],
                mode='bilinear', align_corners=False
            )
            
        f_high = self.proj_high(feat_high)
        f_low  = self.proj_low(feat_low)
        
        f_high = self.attn_high(f_high)
        f_low  = self.attn_low(f_low)
        
        gate = torch.sigmoid(self.gate_conv(torch.cat([f_high, f_low], dim=1)))
        fused = f_low + gate * (f_high - f_low)
        
        refined = self.tagm(fused)
        out = self.mix(refined)
        return out

# -------------------------------------------------------------------------
# HSS-Net Backbone
# -------------------------------------------------------------------------
class Backbone(nn.Module):
    def __init__(self, config, num_blocks=[4, 6, 6, 8], bias=False):
        super().__init__()
        channels, channels_mult = config.backbone.channels, tuple(config.backbone.channels_mult)
        dropout = config.backbone.dropout
        in_channels = 3
        out_channels = 3
        self.ch = channels
        self.ch_level = [self.ch * channels_mult[i] for i in range(len(channels_mult))]
        self.mlp_ratio = 2.00
        
        # Input
        self.conv_in = nn.Conv2d(in_channels, self.ch_level[0], 3, padding=1, bias=bias)
        
        # E1
        self.attn1_1 = nn.Sequential(*[HSSB(self.ch_level[0], 1, self.mlp_ratio, dropout=dropout) for _ in range(num_blocks[0])])
        self.down1_2 = nn.Conv2d(self.ch_level[0], self.ch_level[1], 3, stride=2, padding=1, bias=bias)
        
        # E2
        self.attn2_1 = nn.Sequential(*[HSSB(self.ch_level[1], 2, self.mlp_ratio, dropout=dropout) for _ in range(num_blocks[1])])
        self.down2_3 = nn.Conv2d(self.ch_level[1], self.ch_level[2], 3, stride=2, padding=1, bias=bias)
        
        # E3
        self.attn3_1 = nn.Sequential(*[HSSB(self.ch_level[2], 4, self.mlp_ratio, dropout=dropout) for _ in range(num_blocks[2])])
        self.down3_4 = nn.Conv2d(self.ch_level[2], self.ch_level[3], 3, stride=2, padding=1, bias=bias)
        
        # Middle
        self.attn4 = nn.Sequential(*[HSSB(self.ch_level[3], 8, self.mlp_ratio, dropout=dropout) for _ in range(num_blocks[3])])
        self.up4_3 = nn.ConvTranspose2d(self.ch_level[3], self.ch_level[2], 4, stride=2, padding=1)
        
        # D3
        self.chcat3 = nn.Conv2d(self.ch_level[2]*2, self.ch_level[2], kernel_size=1, bias=bias)
        self.attn3_2 = nn.Sequential(*[HSSB(self.ch_level[2], 4, self.mlp_ratio, dropout=dropout) for _ in range(num_blocks[2])])
        self.up3_2 = nn.ConvTranspose2d(self.ch_level[2], self.ch_level[1], 4, stride=2, padding=1)
        
        # D2
        self.chcat2 = nn.Conv2d(self.ch_level[1]*2, self.ch_level[1], kernel_size=1, bias=bias)
        self.attn2_2 = nn.Sequential(*[HSSB(self.ch_level[1], 2, self.mlp_ratio, dropout=dropout) for _ in range(num_blocks[1])])
        self.up2_1 = nn.ConvTranspose2d(self.ch_level[1], self.ch_level[0], 4, stride=2, padding=1)
        
        # D1
        self.attn1_2 = nn.Sequential(*[HSSB(self.ch_level[0]*2, 1, self.mlp_ratio, dropout=dropout) for _ in range(num_blocks[0])])
        
        # Out
        self.conv_out = nn.Conv2d(self.ch_level[0]*2, out_channels, 3, padding=1, bias=bias)

        # === 跨尺度融合模块 (CFFM) ===
        self.fuse_x4 = CFFM(self.ch_level[0], self.ch_level[1], out_ch=self.ch_level[0], bias=bias)
        self.fuse_x5 = CFFM(self.ch_level[1], self.ch_level[2], out_ch=self.ch_level[1], bias=bias)
        self.fuse_x6 = CFFM(self.ch_level[0], self.ch_level[2], out_ch=self.ch_level[2], bias=bias)

        self.x6_to_s3 = nn.Identity()  
        self.x5_to_s2 = nn.Identity()  
        self.x4_to_s1 = nn.Identity()  

    def forward(self, x):
        # Encode
        x1 = self.conv_in(x)
        x1 = self.attn1_1(x1)  # c0
        x2 = self.down1_2(x1)
        x2 = self.attn2_1(x2)  # c1
        x3 = self.down2_3(x2)
        x3 = self.attn3_1(x3)  # c2
        h  = self.down3_4(x3)  # c3

        # Middle
        h = self.attn4(h)
        h = self.up4_3(h)      

        # === 跨尺度特征融合 (CFFM) ===
        x4 = self.fuse_x4(x1, x2)   
        x5 = self.fuse_x5(x2, x3)   
        x6 = self.fuse_x6(x1, x3)   

        s3 = self.x6_to_s3(F.interpolate(x6, size=x3.shape[-2:], mode='bilinear', align_corners=False))
        s2 = self.x5_to_s2(F.interpolate(x5, size=x2.shape[-2:], mode='bilinear', align_corners=False))
        s1 = self.x4_to_s1(F.interpolate(x4, size=x1.shape[-2:], mode='bilinear', align_corners=False))

        # Decode
        h = self.chcat3(torch.cat([h, s3], dim=1))
        h = self.attn3_2(h)
        h = self.up3_2(h)

        h = self.chcat2(torch.cat([h, s2], dim=1))
        h = self.attn2_2(h)
        h = self.up2_1(h)

        h = self.attn1_2(torch.cat([h, s1], dim=1))
        h = self.conv_out(h)

        return x + h