import torch
from einops import pack, rearrange, repeat, unpack
from einops.layers.torch import Rearrange
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout = 0.1, max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(max_len, 1, dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        seq_len,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0
    ):
        super().__init__()
        assert (seq_len % patch_size) == 0
        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (n p) -> b n (p c)", p=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, series):
        series = series.permute(0, 2, 1)
        x = self.to_patch_embedding(series)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "d -> b d", b=b)
        x, ps = pack([cls_tokens, x], "b * d")
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        cls_tokens, _ = unpack(x, ps, "b * d")
        return self.mlp_head(cls_tokens)


class ConvBackbone(nn.Module):
    def __init__(
        self,
        input_ch,
        transformer_ch,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_ch, transformer_ch, 1),
            nn.GELU(),
            nn.Conv1d(transformer_ch, transformer_ch, 1),
            nn.GELU(),
            nn.Conv1d(transformer_ch, transformer_ch, 1),
            nn.GELU(),
            nn.Conv1d(transformer_ch, transformer_ch, 1),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.input_proj(x)
        return x


class PreConvTransformer(nn.Module):
    def __init__(
        self,
        *,
        hidden_ch,
        num_classes,
        input_dim,
        hidden_dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0
    ):
        super().__init__()
        self.convbackbone = ConvBackbone(input_ch=channels, transformer_ch=hidden_ch)
        self.to_embedding = nn.Linear(input_dim, hidden_dim, bias=False)
        self.pos_embedding = nn.Parameter(torch.randn(1, hidden_ch + 1, hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(hidden_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(hidden_dim, depth, heads, dim_head, mlp_dim, dropout)
        self.mlp_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes))

    def forward(self, series):
        series = series.permute(0, 2, 1)
        x = self.convbackbone(series)
        x = self.to_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "d -> b d", b=b)
        x, ps = pack([cls_tokens, x], "b * d")
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        cls_tokens, _ = unpack(x, ps, "b * d")
        return self.mlp_head(cls_tokens)

class PreConvPositionalEncodingTransformer(nn.Module):
    def __init__(
        self,
        *,
        hidden_ch,
        num_classes,
        input_dim,
        hidden_dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0
    ):
        super().__init__()
        self.convbackbone = ConvBackbone(input_ch=channels, transformer_ch=hidden_ch)
        self.to_embedding = nn.Linear(input_dim, hidden_dim, bias=False)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout, max_len=hidden_ch)
        self.cls_token = nn.Parameter(torch.randn(hidden_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(hidden_dim, depth, heads, dim_head, mlp_dim, dropout)
        self.mlp_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes))

    def forward(self, series):
        series = series.permute(0, 2, 1)
        x = self.convbackbone(series)
        x = self.to_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "d -> b d", b=b)
        x, ps = pack([cls_tokens, x], "b * d")
        print(x.shape)
        print("-----------------------------------")
        x = self.positional_encoding(x)
        x = self.dropout(x)
        x = self.transformer(x)
        cls_tokens, _ = unpack(x, ps, "b * d")
        return self.mlp_head(cls_tokens)