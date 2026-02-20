import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    """
    Compute scaled dot-product attention.
    key, query, value: [bs, num_attention_heads, seq_len, attention_head_size]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    ### YOUR CODE HERE
    # Compute attention scores: Q * K^T / sqrt(d_k)
    # Shape: [bs, num_attention_heads, seq_len, seq_len]
    attention_scores = torch.matmul(query, key.transpose(-1, -2))
    attention_scores = attention_scores / (self.attention_head_size ** 0.5)

    # Create causal mask: lower triangular matrix
    seq_len = query.size(2)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=query.device, dtype=query.dtype))
    # Convert to attention mask format: 0 for valid, large negative for masked
    causal_mask = (1.0 - causal_mask) * -10000.0
    # Expand to [1, 1, seq_len, seq_len] for broadcasting
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    # Apply causal mask
    attention_scores = attention_scores + causal_mask

    # Apply the padding attention mask (mask has 0 for valid positions, large negative for masked)
    attention_scores = attention_scores + attention_mask

    # Normalize the attention scores to probabilities
    attention_probs = nn.functional.softmax(attention_scores, dim=-1)

    # Apply dropout
    attention_probs = self.dropout(attention_probs)

    # Compute context: attention_probs * V
    # Shape: [bs, num_attention_heads, seq_len, attention_head_size]
    context = torch.matmul(attention_probs, value)

    # Reshape back to [bs, seq_len, hidden_state]
    context = rearrange(context, 'b h t d -> b t (h d)')

    return context


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
