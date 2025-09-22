import functools
from typing import Tuple
import torch
import torch.nn as nn
import math

from torch.nn.init import zeros_
from torch.nn.modules.module import T

from torch.nn.functional import scaled_dot_product_attention

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class Embed(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer = None,
            bias: bool = True,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[..., None].float() * freqs[None, ...]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels,):
        embeddings = self.embedding_table(labels)
        return embeddings

class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        out_channels,
        use_mlp: bool = False,
        cond_hidden_size: int | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cond_hidden_size = hidden_size if cond_hidden_size is None else cond_hidden_size
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if use_mlp:
            self.linear = nn.Sequential(
                nn.Linear(hidden_size, hidden_size*4),
                nn.GELU(),
                nn.Linear(hidden_size*4, out_channels),
            )
        else:
            self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(self.cond_hidden_size, 2*hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    def forward(self, x):
        x =  self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
        return x

def precompute_freqs_cis_2d(dim: int, height: int, width:int, theta: float = 10000.0, scale=16.0):

    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)) 
    x_freqs = torch.outer(x_pos, freqs).float() 
    y_freqs = torch.outer(y_pos, freqs).float()
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1) 
    freqs_cis = freqs_cis.reshape(height*width, -1)
    return freqs_cis


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_cis = freqs_cis[None, :, None, :]

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) 
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) 
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = RMSNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, pos, mask) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_emb(q, k, freqs_cis=pos)
        q = q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)  
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()  
        v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()

        x = scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop.p if self.training else 0.0)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(context_dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        B_ctx, M, C_ctx = context.shape

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) 
        kv = self.kv_proj(context).reshape(B_ctx, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) 
        k, v = kv[0], kv[1]

        attn = scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop.p if self.training else 0.0)

        x = attn.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DDTBlock(nn.Module):
    def __init__(self, hidden_size, groups, mlp_ratio=4.0, context_dim=None, is_encoder_block=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = RAttention(hidden_size, num_heads=groups, qkv_bias=False)
        
        self.norm_cross = RMSNorm(hidden_size, eps=1e-6) if context_dim else nn.Identity()
        self.cross_attn = CrossAttention(hidden_size, context_dim, groups) if context_dim else None
        
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)
        
        self.is_encoder_block = is_encoder_block
        if not is_encoder_block:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )

    def forward(self, x, c, pos, mask=None, context=None, shared_adaLN=None):
        if self.is_encoder_block:
            adaLN_output = shared_adaLN(c)
        else:
            adaLN_output = self.adaLN_modulation(c)
            
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaLN_output.chunk(6, dim=-1)
        
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), pos, mask=mask)
        
        if self.cross_attn is not None and context is not None:
            x = x + self.cross_attn(self.norm_cross(x), context=context)

        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DDT(nn.Module):
    def __init__(
            self,
            in_channels=4,
            num_groups=12,
            hidden_size=1152,
            num_blocks=18,
            num_encoder_blocks=4,
            patch_size=2,
            learn_sigma=True,
            deep_supervision=0,
            weight_path=None,
            load_ema=False,
            y_embed_dim=768,
            experiment="baseline",
            num_classes=1000,
            txt_embed_dim=None,
            txt_max_length=None,
            num_text_blocks=None,
            use_cross_attention=False,
            use_tread=False,
            tread_dropout_ratio=0.5,
            tread_prev_blocks=3,
            tread_post_blocks=1,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.num_blocks = num_blocks
        self.num_encoder_blocks = num_encoder_blocks
        self.patch_size = patch_size
        self.x_embedder = Embed(in_channels*patch_size**2, hidden_size, bias=True)
        self.s_embedder = Embed(in_channels*patch_size**2, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        if txt_embed_dim is not None:
            self.y_embedder = nn.Linear(txt_embed_dim, hidden_size, bias=False)
        else:
            self.y_embedder = nn.Linear(y_embed_dim, hidden_size, bias=False)

        patch_elements = in_channels * patch_size**2
        if experiment == "multiple_final_layers_with_skip":
            if patch_size % 4 != 0:
                raise ValueError("multiple_final_layers_with_skip requires patch_size divisible by 4")
            if patch_elements % 16 != 0:
                raise ValueError("Patch elements must be divisible by 16 for multiple_final_layers_with_skip")
            per_head_elements = patch_elements // 16
            self.final_layers = nn.ModuleList([
                FinalLayer(
                    hidden_size * 2,
                    per_head_elements,
                    use_mlp=True,
                    cond_hidden_size=hidden_size,
                )
                for _ in range(16)
            ])
            skip_patch = patch_size // 4
            self.skip_patch_size = skip_patch
            self.skip_embedding = Embed(in_channels * skip_patch**2, hidden_size, bias=True)
        else:
            self.final_layer = FinalLayer(hidden_size, patch_elements)

        self.experiment = experiment

        self.weight_path = weight_path

        self.shared_encoder_adaLN = nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        self.load_ema = load_ema
        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):

            context_dim = self.hidden_size if i % 2 == 0 and i < self.num_encoder_blocks else None

            is_encoder = i < self.num_encoder_blocks
            self.blocks.append(
                DDTBlock(
                    self.hidden_size,
                    self.num_groups,
                    context_dim=context_dim,
                    is_encoder_block=is_encoder 
                )
            )
        self.initialize_weights()
        self.precompute_pos = dict()

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            pos = precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos

    def initialize_weights(self):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        if hasattr(self, "skip_embedding"):
            w = self.skip_embedding.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.skip_embedding.proj.bias, 0)

        nn.init.normal_(self.y_embedder.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.constant_(self.shared_encoder_adaLN[-1].weight, 0)
        nn.init.constant_(self.shared_encoder_adaLN[-1].bias, 0)

        for i in range(self.num_encoder_blocks, self.num_blocks):
            nn.init.constant_(self.blocks[i].adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.blocks[i].adaLN_modulation[-1].bias, 0)

        if hasattr(self, "final_layers"):
            for final_layer in self.final_layers:
                nn.init.constant_(final_layer.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(final_layer.adaLN_modulation[-1].bias, 0)
                if isinstance(final_layer.linear, nn.Sequential):
                    nn.init.constant_(final_layer.linear[-1].weight, 0)
                    nn.init.constant_(final_layer.linear[-1].bias, 0)
                else:
                    nn.init.constant_(final_layer.linear.weight, 0)
                    nn.init.constant_(final_layer.linear.bias, 0)
        else:
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
            if isinstance(self.final_layer.linear, nn.Sequential):
                nn.init.constant_(self.final_layer.linear[-1].weight, 0)
                nn.init.constant_(self.final_layer.linear[-1].bias, 0)
            else:
                nn.init.constant_(self.final_layer.linear.weight, 0)
                nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y, s=None, mask=None):
        B, _, H, W = x.shape
        pos = self.fetch_pos(H//self.patch_size, W//self.patch_size, x.device)
        if self.experiment == "multiple_final_layers_with_skip":
            skip_patch = self.skip_patch_size
            skip_unfolded = torch.nn.functional.unfold(
                x, kernel_size=skip_patch, stride=skip_patch
            ).transpose(1, 2)
            skip_tokens = self.skip_embedding(skip_unfolded)
            skip_h = H // skip_patch
            skip_w = W // skip_patch
            coarse_h = H // self.patch_size
            coarse_w = W // self.patch_size
            sub_factor = self.patch_size // skip_patch
            skip_tokens = skip_tokens.view(B, skip_h, skip_w, self.hidden_size)
            skip_tokens = skip_tokens.view(B, coarse_h, sub_factor, coarse_w, sub_factor, self.hidden_size)
            num_skip_heads = sub_factor * sub_factor
            skip_tokens = skip_tokens.permute(2, 4, 0, 1, 3, 5).reshape(
                num_skip_heads, B, coarse_h * coarse_w, self.hidden_size
            )

        x_unfolded = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        t_emb = self.t_embedder(t.view(-1)).view(B, 1, self.hidden_size)
        y_emb = self.y_embedder(y)

        t_cond = nn.functional.silu(t_emb) 

        if s is None:
            s = self.s_embedder(x_unfolded)
            for i in range(self.num_encoder_blocks):
                block_context = y_emb if i % 2 == 0 else None
                s = self.blocks[i](s, t_cond, pos, mask, context=block_context, shared_adaLN=self.shared_encoder_adaLN)
        
        s_cond = s

        x = self.x_embedder(x_unfolded)
        for i in range(self.num_encoder_blocks, self.num_blocks):

            x = self.blocks[i](x, s_cond, pos, None, shared_adaLN=None)     

        if self.experiment == "multiple_final_layers_with_skip":
            head_outputs = []
            for idx, final_layer in enumerate(self.final_layers):
                skip_features = skip_tokens[idx]
                final_input = torch.cat((x, skip_features), dim=-1)
                head_outputs.append(final_layer(final_input, s_cond))
            x = torch.stack(head_outputs, dim=0)
            x = x.permute(1, 2, 0, 3).contiguous()
            x = x.view(B, x.shape[1], -1)
        else:
            x = self.final_layer(x, s_cond)

        x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), (H, W), kernel_size=self.patch_size, stride=self.patch_size)
        return x
