import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from TeXOCR.utils import exists, get, max_negative_val, topk


class GeGLU(nn.Module):
    """Gated GeLU activation function (arxiv.org/abs/2002.05202)."""
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.fc = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.fc(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class PositionalEmbedding(nn.Module):
    """Standard positonal embedding."""
    def __init__(self, embed_dim: int, max_len: int):
        super().__init__()
        self.embedding = nn.Embedding(max_len, embed_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.shape[1], device=x.device)
        return self.embedding(pos)[None, :, :]
    

class Residual(nn.Module):
    """Residual (skip) connection."""
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return x + residual


class MLP(nn.Module):
    """Transformer block feedforward layer."""
    def __init__(
        self, 
        embed_dim: int, 
        dim_out: int = None, 
        exp_factor: int = 4, 
        glu: bool = True, 
        dropout: float = 0.
    ):
        super().__init__()
        hidden_dim = embed_dim * exp_factor
        dim_out = dim_out or embed_dim

        self.fc_in = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU()
        ) if not glu else GeGLU(embed_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, dim_out)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module (https://arxiv.org/pdf/1706.03762)"""
    def __init__(
        self,
        embed_dim : int,
        heads: int = 8,
        dim_head: int = 64,
        causal: bool = False,
        dropout: float = 0.,
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal

        self.scale = dim_head ** -0.5

        # Queries, keys and values linear transformations
        dim_qk = dim_head * heads
        dim_v = dim_head * heads
        self.q = nn.Linear(embed_dim, dim_qk, bias=False)
        self.k = nn.Linear(embed_dim, dim_qk, bias=False)
        self.v = nn.Linear(embed_dim, dim_v, bias=False)

        # Set up dropout, softmax and final output layer
        self.dropout = nn.Dropout(dropout)
        self.softmax = F.softmax
        self.fc_out = nn.Sequential(
            nn.Linear(dim_v, embed_dim * 2),
            nn.GLU()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        enc: torch.Tensor = None,
        mask: torch.Tensor = None,
        enc_mask: torch.Tensor = None, 
    ) -> torch.Tensor:
        
        B, N, _ = x.shape
        H = self.heads
        device = x.device

        # Existence of enc determines whether cross-attention or self-attention
        if enc is not None:
            kv_input = enc
        else:
            kv_input = x

        # Setting the input to the attention module
        q_input = x
        k_input = kv_input
        v_input = kv_input

        q = self.q(q_input)
        k = self.k(k_input)
        v = self.v(v_input)
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h = H), (q, k, v))

        # Create input mask if one has been provided
        input_mask = None
        if any(map(exists, (mask, enc_mask))):
            
            if mask is not None:
                q_mask = mask
            else:
                q_mask = torch.ones((B, N), device=device).bool()
            
            k_mask = q_mask if not exists(enc) else enc_mask

            if k_mask is None:
                k_mask = torch.ones((B, k.shape[-2]), device=device).bool()

            q_mask = einops.rearrange(q_mask, 'b i -> b () i ()')
            k_mask = einops.rearrange(k_mask, 'b j -> b () () j')
            input_mask = q_mask * k_mask
        
        # Dot product QK^T / \sqrt{dim_head}
        energy = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_val = max_negative_val(energy)

        pre_softmax_attn = energy.clone()

        # Perform masking
        if exists(input_mask):
            energy.masked_fill_(~input_mask, mask_val)
        
        # For causal attention, create triangular mask to ensure no look-ahead
        if self.causal:
            i, j = energy.shape[-2:]
            r = torch.arange(i, device=device)
            mask = einops.rearrange(r, 'i -> () () i ()') < einops.rearrange(r, 'j -> () () () j')
            mask = F.pad(mask, (j-i, 0), value=False)
            energy.masked_fill_(mask, mask_val)
            del mask

        attn = self.softmax(energy, dim=-1)
        post_softmax_attn = attn.clone()

        attn = self.dropout(attn)

        # Dot product with the values
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')

        intermediates = {
            'pre_softmax_attn': pre_softmax_attn,
            'post_softmax_attn': post_softmax_attn
        }

        return self.fc_out(out), intermediates
        

class AttentionLayers(nn.Module):
    """Stack of attention layers."""
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        heads: int = 8,
        causal: bool = False,
        cross_attend: bool = False,
        **ff_kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList([])

        self.cross_attend = cross_attend
        norm = nn.LayerNorm(embed_dim)

        # Setting the default block architecture depending on whether cross-attention present
        if cross_attend:
            block = ('self', 'cross', 'mlp')
        else:
            block = ('self', 'mlp')

        layer_types = block * num_layers
        self.layer_types = layer_types

        # Construct attention layers
        for layer_type in self.layer_types:
            if layer_type == 'self':
                layer = MultiHeadAttention(embed_dim, heads=heads, causal=causal)
            elif layer_type == 'cross':
                layer = MultiHeadAttention(embed_dim, heads=heads)
            elif layer_type == 'mlp':
                layer = MLP(embed_dim, **ff_kwargs)

            residual = Residual()
            self.layers.append(nn.ModuleList([norm, layer, residual]))

    def forward(
        self,
        x: torch.Tensor,
        enc: torch.Tensor = None,
        mask: torch.Tensor = None,
        enc_mask: torch.Tensor = None,
        return_hidden: bool = False
    ) -> torch.Tensor:
        
        if self.cross_attend:
            assert enc is not None, 'Must provide enc if cross_attend is True.'

        hiddens = []
        intermediates = []
        for i, (layer, (norm, block, res)) in enumerate(zip(self.layer_types, self.layers)):
            
            if layer == 'self':
                hiddens.append(x)

            residual = x
            x = norm(x)

            if layer == 'self':
                out, inter = block(x, mask=mask)
            elif layer == 'cross':
                out, inter = block(x, enc=enc, mask=mask, enc_mask=enc_mask)
            elif layer == 'mlp':
                out = block(x)

            x = res(out, residual)

            if layer in ('self', 'cross'):
                intermediates.append(inter)
            
            last = (i == (len(self.layers) - 1))
            if not last:
                x = norm(x)
            
        if return_hidden:
            intermediates = {
                'hiddens': hiddens,
                'attn_intermediates': intermediates
            }

            return x, intermediates
        
        return x


class EncoderLayers(AttentionLayers):
    """Encoder attention layers (no cross attention)."""
    def __init__(self, **kwargs):
        super().__init__(causal=False, **kwargs)


class DecoderLayers(AttentionLayers):
    """Decoder attention layers (includes an encoded input and causal attention)."""
    def __init__(self, **kwargs):
        super().__init__(causal=True, **kwargs)