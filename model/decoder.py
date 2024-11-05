import torch
import torch.nn as nn
import torch.nn.functional as F   

class MLP(nn.Module):
    """Transformer block Multi-Layer Perceptron (MLP)."""
    def __init__(self, embed_dim: int, exp_factor: int, activation: nn.Module = nn.GELU()):
        """
        Args:
            embed_dim: Embedding dimension.
            exp_factor: MLP dimension expansion factor.
            activation: Activation function.
        """
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * exp_factor),
            activation,
            nn.Linear(embed_dim * exp_factor, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)  


class MultiHeadAttention(nn.Module):
    """Multi-headed attention module."""
    def __init__(self, embed_dim: int, heads: int):
        """
        Args:
            embed_dim: Embedding dimension.
            heads: Number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        
        assert (self.head_dim * heads == embed_dim), "Embedding size must be divisible by the number of attention heads."

        # Queries, keys, and values linear transformations
        self.q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.v = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(heads * self.head_dim, embed_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            q: Queries tensor.
            k: Keys tensor.
            v: Values tensor.
            mask: Mask tensor.
        """
        N = q.shape[0] # Batch size
        q_len, k_len, v_len = q.shape[1], k.shape[1], v.shape[1] # Sequence length
        
        # Split the embedding into self.heads pieces
        q = q.reshape(N, q_len, self.heads, self.head_dim)
        k = k.reshape(N, k_len, self.heads, self.head_dim)
        v = v.reshape(N, v_len, self.heads, self.head_dim)
        
        # Apply linear transformations to queries, keys, and values
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        
        # Compute scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [q, k])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = F.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, v]).reshape(
            N, q_len, self.heads * self.head_dim
        )
        
        # Apply linear transformation to the concatenated heads
        out = self.fc_out(out)
        
        return out
    

class TransformerBlock(nn.Module):
    """Transformer block module."""
    def __init__(self, embed_dim: int, heads: int, exp_factor: int, dropout: float):
        """
        Args:
            embed_dim: Embedding dimension.
            heads: Number of attention heads.
            exp_factor: MLP dimension expansion factor.
            dropout: Dropout rate.
        """
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = MLP(embed_dim, exp_factor, activation=nn.GELU())
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: Queries tensor.
            k: Keys tensor.
            v: Values tensor.
            mask: Mask tensor.
        """
        attention = self.attention(q, k, v, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + q))
        forward = self.mlp(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    

class TransformerEncoder(nn.Module):
    """Transformer encoder module."""
    def __init__(
            self,
            src_vocab_size: int,
            embed_dim: int,
            num_layers: int,
            heads: int,
            device: torch.device,
            exp_factor: int,
            dropout: float,
            max_length: int,
    ):
        """
        Args:
            src_vocab_size: Source vocabulary size.
            embed_dim: Embedding dimension.
            num_layers: Number of layers.
            heads: Number of attention heads.
            device: Device.
            exp_factor: MLP dimension expansion factor.
            dropout: Dropout rate.
            max_length: Maximum sequence length.
        """
        super(TransformerEncoder, self).__init__()
        self.embed_size = embed_dim
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    heads,
                    dropout=dropout,
                    exp_factor=exp_factor,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor.
            mask: Mask tensor.
        """
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        print(out.shape)

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out
    

class DecoderBlock(nn.Module):
    """Transformer decoder block."""
    def __init__(self, embed_dim: int, heads: int, exp_factor: int, dropout: float, device: torch.device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, heads)
        self.transformer_block = TransformerBlock(
            embed_dim, heads, exp_factor, dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor, src_mask: torch.Tensor, trg_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor.
            key: Keys.
            value: Values.
            src_mask: Source mask.
            trg_mask: Target mask.
        """
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(query, k, v, src_mask)
        return out
    

class TransformerDecoder(nn.Module):
    """Transformer decoder module."""
    def __init__(
            self,
            vocab_size: int,
            embed_dim: int,
            num_layers: int,
            heads: int,
            exp_factor: int,
            dropout: float,
            device: torch.device,
            max_length: int,
    ):
        """
        Args:
            trg_vocab_size: Target vocabulary size.
            embed_dim: Embedding dimension.
            num_layers: Number of layers.
            heads: Number of attention heads.
            exp_factor: MLP dimension expansion factor.
            dropout: Dropout rate.
            device: Device.
            max_length: Maximum sequence length.
        """
        super(TransformerDecoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_dim, heads, exp_factor, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor, trg_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor.
            enc_out: Encoder output (used as keys and values).
            src_mask: Source mask.
            trg_mask: Target mask.
        """
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out
    
    def generate(self, start_token, max_length, eos_token):
        """
        Args:
            start_token: Start token.
            max_length: Maximum length.
            eos_token: End of sequence token.
        """

        trg_tensor = torch.tensor([start_token]).unsqueeze(0).to(self.device)
        for i in range(max_length):
            trg_mask = self.make_trg_mask(trg_tensor)
            out = self.forward(trg_tensor, trg_mask)
            out = out.argmax(2)[:, -1].item()
            trg_tensor = torch.cat((trg_tensor, torch.tensor([[out]]).to(self.device)), dim=1)
            if out == eos_token:
                break

        return trg_tensor
        

    

class Transformer(nn.Module):
    """Transformer sequence-to-sequence model."""
    def __init__(
            self,
            src_vocab_size: int,
            trg_vocab_size: int,
            src_pad_idx: int,
            trg_pad_idx: int,
            embed_dim: int = 512,
            num_layers: int = 6,
            exp_factor: int = 4,
            heads: int = 8,
            dropout: float = 0.,
            device: torch.device = "cpu",
            max_length: int = 100,
    ):
        """
        Args:
            src_vocab_size: Source vocabulary size.
            trg_vocab_size: Target vocabulary size.
            src_pad_idx: Source padding index.
            trg_pad_idx: Target padding index.
            embed_dim: Embedding dimension.
            num_layers: Number of layers.
            exp_factor: MLP dimension expansion factor.
            heads: Number of attention heads.
            dropout: Dropout rate.
            device: Device.
            max_length: Maximum sequence length.
        """

        super(Transformer, self).__init__()

        self.encoder = TransformerEncoder(
            src_vocab_size,
            embed_dim,
            num_layers,
            heads,
            device,
            exp_factor,
            dropout,
            max_length,
        )

        self.decoder = TransformerDecoder(
            trg_vocab_size,
            embed_dim,
            num_layers,
            heads,
            exp_factor,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Source tensor.
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trg: Target tensor.
        """
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Source tensor.
            trg: Target tensor.
        """
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
    

if __name__ == '__main__':
    device = "cpu"
    x = torch.tensor([[1, 5, 6, 4, 4], [1, 8, 7, 3, 4]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 0, 0, 0, 0]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
    # print(trg[:, :-1])
    out = model(x, trg[:, :-1])
    # print(out.shape)
    # print(out)