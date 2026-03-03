'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW

TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # Fine-tuning strategy: to prevent overfitting on the small sonnet dataset,
    # freeze the lower transformer layers and only fine-tune the top few layers,
    # the final layer norm, and the word/position embeddings.
    # Controlled by args.finetune_layers (default: 4 = unfreeze top 4 of 12 layers).
    num_layers = len(self.gpt.gpt_layers)
    finetune_layers = getattr(args, 'finetune_layers', num_layers)

    # First, freeze everything.
    for param in self.gpt.parameters():
      param.requires_grad = False

    # Unfreeze the top `finetune_layers` transformer layers.
    for i in range(max(0, num_layers - finetune_layers), num_layers):
      for param in self.gpt.gpt_layers[i].parameters():
        param.requires_grad = True

    # Always unfreeze the final layer norm (critical for generation quality).
    for param in self.gpt.final_layer_norm.parameters():
      param.requires_grad = True

    # Always unfreeze embeddings (needed for domain adaptation to sonnets).
    for param in self.gpt.word_embedding.parameters():
      param.requires_grad = True
    for param in self.gpt.pos_embedding.parameters():
      param.requires_grad = True

    trainable = sum(p.numel() for p in self.gpt.parameters() if p.requires_grad)
    total = sum(p.numel() for p in self.gpt.parameters())
    print(f"Fine-tuning {trainable}/{total} parameters "
          f"(top {finetune_layers}/{num_layers} layers + embeddings + final LN).")

  def forward(self, input_ids, attention_mask):
    """
    This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
    not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
    not just the distribution over next tokens for the last token!
    """
    ### YOUR CODE HERE
    # Get output dict from GPT-2
    output = self.gpt(input_ids, attention_mask)

    # Extract hidden states from the dict — adjust key if needed
    hidden_states = output['last_hidden_state']

    # Project hidden states to vocabulary logits using weight tying
    logits = torch.matmul(hidden_states, self.gpt.word_embedding.weight.T)

    return logits


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  def _apply_repetition_penalty(self, logits, token_ids, penalty=1.2):
    """Apply repetition penalty to discourage repeated tokens (Keskar et al., 2019)."""
    for token_id in set(token_ids[0].tolist()):
      if logits[0, token_id] > 0:
        logits[0, token_id] /= penalty
      else:
        logits[0, token_id] *= penalty
    return logits

  def _apply_top_k(self, logits, top_k=50):
    """Zero out all logits below the top-k highest values."""
    if top_k > 0:
      top_k = min(top_k, logits.size(-1))
      indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
      logits[indices_to_remove] = float('-inf')
    return logits

  def _apply_top_p(self, logits, top_p=0.9):
    """Apply nucleus (top-p) filtering to logits."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift so that the first token above threshold is kept
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Scatter sorted tensors back to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
      dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits[indices_to_remove] = float('-inf')
    return logits

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, top_k=50,
               repetition_penalty=1.2, num_beams=1, max_length=128):
    """
    Generates an original sonnet with improved decoding strategies inspired by
    HuggingFace's model.generate().

    Supports:
      - Temperature-scaled sampling
      - Top-k filtering (limits sampling to k most likely tokens)
      - Top-p / nucleus sampling (limits sampling to smallest set with cumulative prob >= p)
      - Repetition penalty (penalizes already-generated tokens)
      - Beam search (maintains num_beams hypotheses and selects the best)

    Args:
      encoding: input token ids tensor
      temperature: softmax temperature (higher = more random)
      top_p: cumulative probability threshold for nucleus sampling
      top_k: number of highest probability tokens to keep (0 = disabled)
      repetition_penalty: penalty factor for repeated tokens (1.0 = no penalty)
      num_beams: number of beams for beam search (1 = no beam search, greedy/sampling)
      max_length: maximum number of tokens to generate
    """
    token_ids = encoding.to(self.get_device())

    if num_beams > 1:
      return self._beam_search(token_ids, num_beams, temperature, top_k, top_p,
                                repetition_penalty, max_length)
    else:
      return self._sample(token_ids, temperature, top_k, top_p,
                          repetition_penalty, max_length)

  @torch.no_grad()
  def _sample(self, token_ids, temperature, top_k, top_p, repetition_penalty, max_length):
    """Sampling-based generation with top-k, top-p, and repetition penalty."""
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

    for _ in range(max_length):
      logits_sequence = self.forward(token_ids, attention_mask)
      logits = logits_sequence[:, -1, :]

      # Apply repetition penalty before temperature scaling
      if repetition_penalty != 1.0:
        logits = self._apply_repetition_penalty(logits, token_ids, repetition_penalty)

      # Apply temperature scaling
      logits = logits / temperature

      # Apply top-k filtering
      logits = self._apply_top_k(logits, top_k)

      # Apply top-p (nucleus) filtering
      logits = self._apply_top_p(logits, top_p)

      # Sample from the filtered distribution
      probs = F.softmax(logits, dim=-1)
      sampled_token = torch.multinomial(probs, 1)

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output

  @torch.no_grad()
  def _beam_search(self, token_ids, num_beams, temperature, top_k, top_p,
                   repetition_penalty, max_length):
    """
    Beam search: maintain num_beams hypotheses at each step and pick the
    highest-scoring complete sequence at the end.
    """
    device = self.get_device()
    # Each beam: (log_prob, token_ids_tensor)
    beams = [(0.0, token_ids)]

    for _ in range(max_length):
      all_candidates = []
      all_done = True

      for score, beam_ids in beams:
        # Check if this beam already ended
        if beam_ids[0, -1].item() == self.tokenizer.eos_token_id:
          all_candidates.append((score, beam_ids))
          continue

        all_done = False
        attention_mask = torch.ones(beam_ids.shape, dtype=torch.int64).to(device)
        logits_sequence = self.forward(beam_ids, attention_mask)
        logits = logits_sequence[:, -1, :]

        # Apply repetition penalty
        if repetition_penalty != 1.0:
          logits = self._apply_repetition_penalty(logits, beam_ids, repetition_penalty)

        logits = logits / temperature

        log_probs = F.log_softmax(logits, dim=-1)

        # Select top-k candidates per beam to expand
        topk_log_probs, topk_ids = torch.topk(log_probs, num_beams * 2, dim=-1)

        for i in range(topk_ids.size(-1)):
          next_token = topk_ids[0, i].unsqueeze(0).unsqueeze(0)
          new_score = score + topk_log_probs[0, i].item()
          new_ids = torch.cat([beam_ids, next_token], dim=1)
          all_candidates.append((new_score, new_ids))

      if all_done:
        break

      # Keep top num_beams candidates (length-normalized score)
      all_candidates.sort(key=lambda x: x[0] / max(1, x[1].size(1)), reverse=True)
      beams = all_candidates[:num_beams]

    # Select the best beam (length-normalized)
    best_score, best_ids = max(beams, key=lambda x: x[0] / max(1, x[1].size(1)))

    # Remove trailing EOS if present
    if best_ids[0, -1].item() == self.tokenizer.eos_token_id:
      best_ids = best_ids[:, :-1]

    generated_output = self.tokenizer.decode(best_ids[0].cpu().numpy().tolist())[3:]
    return best_ids, generated_output


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def compute_val_loss(model, val_dataloader, device):
  """Compute the average cross-entropy loss on a validation set."""
  model.eval()
  total_loss = 0
  num_batches = 0
  with torch.no_grad():
    for batch in val_dataloader:
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
      labels = b_ids[:, 1:].contiguous().flatten()
      loss = F.cross_entropy(logits, labels, reduction='mean')

      total_loss += loss.item()
      num_batches += 1

  return total_loss / max(num_batches, 1)


def train(args):
  """Train GPT-2 for sonnet generation."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  # Create a dev set for validation loss (full sonnets for computing perplexity).
  dev_sonnet_dataset = SonnetsDataset(args.dev_sonnet_path)
  dev_dataloader = DataLoader(dev_sonnet_dataset, shuffle=False, batch_size=args.batch_size,
                              collate_fn=dev_sonnet_dataset.collate_fn)

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)

  # Early stopping state.
  best_val_loss = float('inf')
  patience_counter = 0
  best_epoch = -1

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # Ignore the last prediction in the sequence.
      labels = b_ids[:, 1:].contiguous().flatten()  # Ignore the first token to compose the labels.
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches

    # Compute validation loss for early stopping.
    val_loss = compute_val_loss(model, dev_dataloader, device)
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, val loss :: {val_loss :.3f}.")

    print('Generating several output sonnets...')
    model.eval()
    for batch in held_out_sonnet_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p,
                              top_k=args.top_k, repetition_penalty=args.repetition_penalty,
                              num_beams=args.num_beams)
      print(f'{batch[1]}{output[1]}\n\n')

    save_model(model, optimizer, args, f'{epoch}_{args.filepath}')

    # Early stopping: save best model and stop if no improvement for `patience` epochs.
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      patience_counter = 0
      best_epoch = epoch
      save_model(model, optimizer, args, f'best_{args.filepath}')
      print(f"  New best val loss: {best_val_loss:.3f}. Saving best model.")
    else:
      patience_counter += 1
      print(f"  No improvement for {patience_counter} epoch(s) (best val loss: {best_val_loss:.3f} at epoch {best_epoch}).")
      if patience_counter >= args.patience:
        print(f"Early stopping triggered after {epoch + 1} epochs (patience={args.patience}).")
        break


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  # Prefer loading the best model (from early stopping); fall back to the last epoch.
  import os
  best_path = f'best_{args.filepath}'
  last_path = f'{args.epochs-1}_{args.filepath}'
  load_path = best_path if os.path.exists(best_path) else last_path
  print(f"Loading model from {load_path}")
  saved = torch.load(load_path, weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p,
                            top_k=args.top_k, repetition_penalty=args.repetition_penalty,
                            num_beams=args.num_beams)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--dev_sonnet_path", type=str, default="data/TRUE_sonnets_held_out_dev.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--patience", type=int, help="Early stopping patience (epochs without improvement).", default=3)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)
  parser.add_argument("--top_k", type=int, help="Top-k filtering: keep only top k tokens with highest probability.",
                      default=50)
  parser.add_argument("--repetition_penalty", type=float,
                      help="Repetition penalty factor (1.0 = no penalty).", default=1.2)
  parser.add_argument("--num_beams", type=int, help="Number of beams for beam search (1 = no beam search).",
                      default=1)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--finetune_layers", type=int,
                      help="Number of top transformer layers to fine-tune (rest are frozen). "
                           "Use 12 to fine-tune all layers (gpt2 has 12).", default=4)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  generate_submission_sonnets(args)