"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier


Originally forked from Andrej Karpathy's minGPT.

CS224N 2022-23: Homework 5

John Hewitt <johnhew@stanford.edu>
Ansh Khurana <anshk@stanford.edu>
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

import attention

torch.manual_seed(1)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    perceiver = False
    bottleneck_dim = None

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = attention.CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DownProjectBlock(nn.Module):
    """Transformer block used for down projection.
    
    Initialize similarly to the regular tranformer Block class,
    while using the CausalCrossAttention layer instead of the regular
    CausalSelfAttention layer.
    
    You also need to initialize the parameter for the basis vectors `self.C` here.
    Initialize `self.C` with appropriate dimensions and xavier_uniform initalization.
    
    self.C should be 1 x bottleneck_dim x n_embd. We need the first dimension 
    for appropriate broadcasting along the batch_size dimension of the input 
    sequence.
    
    `self.C` will be used to compute the Query vector for the cross attention
    layer.
    """
    def __init__(self, config):
        super().__init__()
        ### YOUR CODE HERE
        ### Hint: Copy over the code from Block and make necessary modifications.
        pass
        ### END YOUR CODE

    def forward(self, x_input):
        """Hint: perform cross-attention between x_input and self.C.
        Use the layernorm layers on C, and then on the input to the MLP.
        """
        ### YOUR CODE HERE
        ### Hint: Copy over the code from Block and make necessary modifications.
        ### Should be around 3-5 lines.
        pass
        ### END YOUR CODE
    
    
class UpProjectBlock(nn.Module):
    """Transformer block used for up projection.
    
    Initialize similarly to the regular transformer Block class,
    while using the CausalCrossAttention layer instead of the regular
    CausalSelfAttention layer.
    """
    def __init__(self, config):
        super().__init__()
        ### YOUR CODE HERE
        ### Hint: Copy over the code from Block and make necessary modifications.
        pass
        ### END YOUR CODE
    
    def forward(self, y, x_input):
        """Hint: perform cross-attention between previous layer's output y and
        x_input. 
        Use the layernorm layers on y, and then on the input to the MLP.
        """
        ### YOUR CODE HERE
        ### Hint: Copy over the code from Block and make necessary modifications.
        ### Should be around 3-5 lines.
        pass
        ### END YOUR CODE
    

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.perceiver = config.perceiver
        if config.perceiver:            
            input_block_size = config.block_size
            
            # input sequence based causal mask
            self.down_block = DownProjectBlock(config)
            
            # bottleneck basis based causal mask
            config.block_size = config.bottleneck_dim
            self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer-2)])
            
            # reset value of the block size back to the original.
            config.block_size = input_block_size
            self.up_block = UpProjectBlock(config)
            
            
        else:
            self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        print("number of parameters: {}".format(sum(p.numel() for p in self.parameters())))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size (%d, %d) is exhausted." % (t, self.block_size)

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x_input = self.drop(token_embeddings + position_embeddings)
        
        if self.perceiver:
            x = self.down_block(x_input)
        else:
            x = x_input
        
        # always compute through the blocks
        x = self.blocks(x)
        
        if self.perceiver:
            x = self.up_block(x, x_input)
            
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)

        return logits, loss

