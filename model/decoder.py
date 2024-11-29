import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from TeXOCR.utils import exists, get, max_negative_val, topk
from TeXOCR.model.attention import AttentionLayers, PositionalEmbedding, DecoderLayers


class Transformer(nn.Module):
    """Transformer module (https://arxiv.org/pdf/1706.03762)"""
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        attn_layers: AttentionLayers,
        dropout: float = 0.,
    ):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), 'Unrecognised class for attn_layers.'

        # Embedding dim and max sequence length
        embed_dim = attn_layers.embed_dim
        self.max_len = max_len

        # Transformer modules: token/pos embeddings, dropout, normalisation, attention layers
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEmbedding(embed_dim, max_len)
        self.embed_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn_layers = attn_layers

        self.init_weights() # Initialise token embedding weights

        # Projection to logits
        self.to_logits = nn.Linear(embed_dim, vocab_size)

    def init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_embeddings: bool = False,
        return_attn: bool = False,
        **kwargs   
    ) -> torch.Tensor:
        
        # Sum the token and position embeddings, apply dropout
        x = self.token_embedding(x)
        x = x + self.pos_embedding(x)
        x = self.embed_dropout(x)

        # Pass through attention layers and normalise
        x, intermediates = self.attn_layers(x, mask=mask, return_hidden=True, **kwargs)
        x = self.norm(x)

        # Convert to logits
        out = self.to_logits(x) if not return_embeddings else x
        
        # Check if attention maps should be returned
        if return_attn:
            attn_maps = list(map(lambda t: t['post_softmax_attn'], intermediates['attn_intermediates']))
            return out, attn_maps

        return out


class AutoRegressiveDecoder(nn.Module):
    """Auto-regressive transformer decoder for sequence generation."""
    def __init__(self, net: Transformer):
        super().__init__()
        self.net = net
        self.max_len = net.max_len
    
    @torch.no_grad()
    def generate(
        self, 
        start_tokens: torch.Tensor, 
        eos_tok: int, 
        max_len: int, 
        temp: float = 1.,
        **kwargs
    ) -> torch.Tensor:
        
        tr, device = self.net.training, start_tokens.device
        start_tokens = start_tokens[None, :] if start_tokens.ndim == 1 else start_tokens
        B, T = start_tokens.shape # Get batch size and sequence length

        self.net.eval()
        output = start_tokens

        # Set mask to True for all tokens if not provided
        mask = kwargs.pop('mask', torch.full_like(output, True, device=device, dtype=torch.bool))

        for i in range(max_len):
            print(i)
            x = output[:, -self.max_len:]
            mask = mask[:, -self.max_len:]

            # Get logits for last token
            logits = self.net(x, mask=mask, **kwargs)[:, -1, :] 
            topk_logits = topk(logits)

            # Sample next tokens
            probs = F.softmax(topk_logits / temp, dim=-1)
            next_tokens = torch.multinomial(probs, 1)

            # Concatenate to output and update mask
            output = torch.cat((output, next_tokens), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

            # Break if all sequences have reached eos_token (CHECK THIS)
            if eos_tok is not None and (output == eos_tok).any(dim=1).all():
                break
        
        generated_tokens = output[:, T:]
        generated_tokens = generated_tokens.squeeze(0) if start_tokens.ndim == 1 else generated_tokens
        
        self.net.train(tr)
        return generated_tokens

    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor = None, 
        return_out: bool = False, 
        **kwargs
    ) -> torch.Tensor:
        
        # Split x into input and output sequences (offset by one)
        x_in = x[:, :-1]
        x_out = x[:, 1:]

        if mask is not None and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]

        out = self.net(x_in, mask=mask, **kwargs)
        loss = F.cross_entropy(out.transpose(1, 2), x_out)

        if return_out:
            return loss, out

        return loss
    

def create_decoder(config: dict):
    """Create a TransformerDecoder from a config dict."""
    assert 'max_length' in config, 'max_length not loaded into config file!'
    assert 'vocab_size' in config, 'vocab_size not loaded into config file!'

    max_length = config['max_length']
    vocab_size = config['vocab_size']
    decoder_args = config['decoder']
    ff_kwargs = {
        'glu': config['glu'],
        'exp_factor': decoder_args['exp_factor']
    }
    dec_layers = DecoderLayers(
        embed_dim=decoder_args['embed_dim'],
        num_layers=decoder_args['num_layers'],
        heads=decoder_args['heads'],
        cross_attend=decoder_args['cross_attend'],
        **ff_kwargs
    )
    transformer = Transformer(
        vocab_size=vocab_size,
        max_len=max_length,
        attn_layers=dec_layers,
        dropout=decoder_args['dropout']
    )
    return AutoRegressiveDecoder(net=transformer)

# def create_decoder(config: dict):
#     """Create a TransformerDecoder from a config dict."""
#     assert 'max_length' in config, 'max_length not loaded into config file!'
#     assert 'vocab_size' in config, 'vocab_size not loaded into config file!'

#     # device = torch.device(config['device'])
#     max_length = config['max_length']
#     vocab_size = config['vocab_size']
#     decoder_args = {
#         'attn_on_attn': True,
#         'cross_attend': True,
#         'ff_glu': True,
#         'rel_pos_bias': False,
#         'use_scalenorm': False
#     }

#     return CustomARWrapper(
#         XTransformerWrapper(
#             num_tokens=vocab_size,
#             max_seq_len=max_length,
#             attn_layers=XDecoder(
#                 dim=256,
#                 depth=4,
#                 heads=8,
#                 **decoder_args
#             )),
#         pad_value=999,
#         ignore_index=999
#         )