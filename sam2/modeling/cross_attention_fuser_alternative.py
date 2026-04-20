import torch 
import torch.nn as nn

class AttentionFuserBlock(nn.Module):

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, ffn_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.1))  # Scaling factor for the cross-attention output before adding to SAM features
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.gate = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        sa_out, _ = self.self_attn(q, q, q)
        q = self.norm1(q + sa_out)
        ca_out, _ = self.cross_attn(q, kv, kv)
        gate = torch.sigmoid(self.gate(torch.cat([q, ca_out], dim=-1)))
        q = self.norm2(q + self.alpha * gate * ca_out)
        q = self.norm3(q + self.ffn(q))
        return q


class CrossAttentionFuser(nn.Module):

    def __init__(
        self, 
        embed_dim: int = 256, 
        num_heads: int = 8,
        ffn_dim: int = 1024, 
        num_blocks: int = 1,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            AttentionFuserBlock(embed_dim, num_heads, ffn_dim, dropout) 
            for _ in range(num_blocks)
        ])

    def forward(self, sam_features: torch.Tensor, dino_features: torch.Tensor) -> torch.Tensor:
        B, C, H, W = sam_features.shape
        q = sam_features.flatten(2).permute(0, 2, 1)
        for block in self.blocks:
            q = block(q, dino_features)
        return q.permute(0, 2, 1).reshape(B, C, H, W)