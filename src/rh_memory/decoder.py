import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        inv_freq = self.inv_freq
        assert isinstance(inv_freq, torch.Tensor)
        t = torch.arange(seq_len, device=device).to(dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return torch.cos(emb), torch.sin(emb)

def apply_rotary_pos_emb(x, cos, sin):
    # x shape: [B, n_heads, seq_len, head_dim]
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    
    # RoPE interlacing (or concatenation depending on convention, we use concat here)
    rotated = torch.cat((-x2, x1), dim=-1)
    
    # Broadcast to match [B, n_heads, seq_len, head_dim]
    return (x * cos.unsqueeze(0).unsqueeze(0)) + (rotated * sin.unsqueeze(0).unsqueeze(0))

class RoPEMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is [B, seq_len (C), embed_dim]
        B, C, _ = x.shape

        q = self.q_proj(x).view(B, C, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, C, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, C, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE mapped to the C bucket size
        cos, sin = self.rotary_emb(C, x.device)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, C, self.embed_dim)
        return self.out_proj(context)

class RoPETransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # MHA block (Pre-Norm)
        src2 = self.norm1(src)
        src2 = self.self_attn(src2)
        src = src + self.dropout1(src2)
        
        # FFN block (Pre-Norm)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

class RHDecoder(nn.Module):
    def __init__(self, sequence_length, bucket_count, d_model=256, n_heads=8, num_layers=4, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        self.sequence_length = sequence_length
        self.bucket_count = bucket_count
        self.d_model = d_model
        
        # Setup continuous projection: [B, C, 3] -> [B, C, D_model]
        self.input_proj = nn.Linear(3, d_model)
        
        # Deep N-layer Transformer backbone with continuous RoPE
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(d_model, n_heads, dim_feedforward=dim_feedforward, dropout=dropout) 
            for _ in range(num_layers)
        ])
        
        # Dense head projecting back out to sequence domain: [B, C, D_model] -> [B, C, n]
        self.output_proj = nn.Linear(d_model, sequence_length)

    def forward(self, tokens_3d: Float[Tensor, "batch C 3"]) -> Float[Tensor, "batch C n"]:
        """
        Forward pass for the decoder backbone.
        tokens_3d: float tensor shape [batch_size, C, 3] encapsulating:
            1. signed_value (sign * magnitude)
            2. gamma
            3. DIB
        """
        x = self.input_proj(tokens_3d)
        
        for layer in self.layers:
            x = layer(x)
            
        logits = self.output_proj(x)
        return logits

    def reconstruct(self, logits: Float[Tensor, "batch C n"]) -> Float[Tensor, "batch n"]:
        """
        Reconstructs the original [B, n] sequence using max extraction per bucket
        and reducing over collisions with scatter_add.
        logits: FloatTensor of shape [batch_size, C, n]
        """
        batch_size, C, n = logits.shape
        
        # 1. Determine single most likely position for each bucket (max over dim=2)
        # predicted_indices shape: [batch_size, C]
        max_logits, predicted_indices = torch.max(logits, dim=2)
        
        # 2. Project back via scatter_add onto a neutral [batch_size, n] tensor
        reconstructed = torch.zeros(batch_size, n, device=logits.device, dtype=logits.dtype)
        reconstructed.scatter_add_(1, predicted_indices, max_logits)
        
        return reconstructed

class RHLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: Float[Tensor, "batch C n"], targets: Float[Tensor, "batch C n"], magnitudes: Float[Tensor, "batch C"]) -> Float[Tensor, ""]:
        """
        logits: FloatTensor [batch_size, C, n]
        targets: FloatTensor [batch_size, C, n] - sparse ground truth masks of `1.0` vs `0.0`.
        magnitudes: FloatTensor [batch_size, C] - representing salience weight.
        """
        B, C, n = logits.shape
        
        # Unreduced BCE loss evaluated per element
        bce_loss_raw = F.binary_cross_entropy_with_logits(
            logits, 
            targets, 
            reduction='none'
        )
        
        # BCE must be sample-weighted strictly by source magnitude.
        # Shape broadcast: [B, C, n] * [B, C, 1]
        weighted_loss = bce_loss_raw * magnitudes.unsqueeze(2)
        
        # Final reduction strategy (average over batch * sequence * buckets)
        return weighted_loss.mean()
