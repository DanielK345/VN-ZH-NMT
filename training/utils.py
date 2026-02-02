"""Training utilities: loss functions, schedulers, and decoding."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from typing import List, Optional
from dataclasses import dataclass
from tqdm import tqdm

try:
    import sacrebleu
except ImportError:
    sacrebleu = None

from .config import RopeConfig


class LabelSmoothedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with label smoothing."""

    def __init__(self, smoothing: float = 0.1, ignore_index: int = 0):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        mask = targets != self.ignore_index
        
        with torch.no_grad():
            true_dist = torch.full_like(
                log_probs, self.smoothing / (vocab_size - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
            true_dist[targets == self.ignore_index] = 0.0
        
        loss = -(true_dist * log_probs).sum(dim=-1)
        loss = loss.masked_fill(~mask, 0.0)
        return loss.sum() / mask.sum().clamp(min=1)


class WarmupInverseSqrtScheduler:
    """Learning rate scheduler with warmup and inverse sqrt decay."""

    def __init__(self, optimizer, warmup_steps: int, lr_base: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.lr_base = lr_base
        self.step_num = 0

    def step(self):
        """Update learning rate."""
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.lr_base * self.step_num / self.warmup_steps
        else:
            lr = self.lr_base * math.sqrt(self.warmup_steps) / math.sqrt(self.step_num)
        
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


@dataclass
class BeamSearchHypothesis:
    """Hypothesis for beam search decoding."""
    tokens: list
    log_prob: float


@torch.no_grad()
def greedy_decode(
    model,
    src_ids: torch.Tensor,
    sp_model: spm.SentencePieceProcessor,
    config: RopeConfig,
    max_len: int = 32,
) -> List[str]:
    """
    Fast greedy decoding for validation.
    
    Args:
        model: Transformer model
        src_ids: Source token IDs
        sp_model: SentencePiece tokenizer
        config: Configuration
        max_len: Maximum decoding length
        
    Returns:
        List of decoded strings
    """
    model.eval()
    batch_size = src_ids.size(0)
    device = src_ids.device
    
    bos = sp_model.piece_to_id(config.bos_token)
    eos = sp_model.piece_to_id(config.eos_token)
    pad = sp_model.piece_to_id(config.pad_token)

    # Encode source
    src_pad_mask = (src_ids == pad)
    src_emb = model.embedding(src_ids) * model.emb_scale
    src_input = model.emb_dropout(src_emb)
    enc_out = src_input
    for layer in model.encoder_layers:
        enc_out = layer(enc_out, src_pad_mask)
    enc_out = model.encoder_final_ln(enc_out)

    # Initialize decoder input
    tgt_ids = torch.full((batch_size, 1), bos, dtype=torch.long, device=device)

    for _ in range(max_len):
        tgt_pad = (tgt_ids == pad)
        tgt_len = tgt_ids.size(1)
        tgt_causal = torch.triu(
            torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device),
            diagonal=1
        )

        tgt_emb = model.embedding(tgt_ids) * model.emb_scale
        tgt_input = model.emb_dropout(tgt_emb)
        dec_out = tgt_input
        for layer in model.decoder_layers:
            dec_out = layer(dec_out, enc_out, tgt_pad, tgt_causal, src_pad_mask)
        dec_out = model.decoder_final_ln(dec_out)

        logits = F.linear(dec_out, model.embedding.weight, model.output_bias)
        next_tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tgt_ids = torch.cat([tgt_ids, next_tokens], dim=1)

        if (next_tokens.squeeze(-1) == eos).all():
            break

    # Decode to text
    decoded = []
    for seq in tgt_ids:
        tokens = seq[1:].cpu().tolist()  # Skip BOS
        if eos in tokens:
            tokens = tokens[:tokens.index(eos)]
        decoded.append(sp_model.decode(tokens))

    return decoded


@torch.no_grad()
def beam_search_decode(
    model,
    src_ids: torch.Tensor,
    sp_model: spm.SentencePieceProcessor,
    config: RopeConfig,
    beam_size: int = 3,
    top_k: int = 3,
    max_len: int = 32,
    length_penalty: float = 0.6,
) -> List[str]:
    """
    Beam search decoding with length penalty.
    
    Args:
        model: Transformer model
        src_ids: Source token IDs
        sp_model: SentencePiece tokenizer
        config: Configuration
        beam_size: Beam width
        top_k: Top-k filtering
        max_len: Maximum decoding length
        length_penalty: Length penalty coefficient
        
    Returns:
        List of decoded strings
    """
    model.eval()
    batch_size = src_ids.size(0)
    device = src_ids.device
    
    bos = sp_model.piece_to_id(config.bos_token)
    eos = sp_model.piece_to_id(config.eos_token)
    pad = sp_model.piece_to_id(config.pad_token)
    
    src_pad_mask = (src_ids == pad)
    src_emb = model.embedding(src_ids) * model.emb_scale
    src_input = model.emb_dropout(src_emb)
    enc_out = src_input
    for layer in model.encoder_layers:
        enc_out = layer(enc_out, src_pad_mask)
    enc_out = model.encoder_final_ln(enc_out)
    
    decoded = []
    for b in range(batch_size):
        beams = [BeamSearchHypothesis(tokens=[bos], log_prob=0.0)]
        finished = []
        single_enc = enc_out[b : b + 1]
        single_mask = src_pad_mask[b : b + 1]
        
        for _ in range(max_len):
            new_beams = []
            for beam in beams:
                if beam.tokens[-1] == eos:
                    finished.append(beam)
                    continue
                
                tgt_ids = torch.tensor(beam.tokens, dtype=torch.long, device=device).unsqueeze(0)
                tgt_pad = (tgt_ids == pad)
                tgt_len = tgt_ids.size(1)
                tgt_causal = torch.triu(
                    torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device),
                    diagonal=1
                )
                
                tgt_emb = model.embedding(tgt_ids) * model.emb_scale
                tgt_input = model.emb_dropout(tgt_emb)
                dec_out = tgt_input
                for layer in model.decoder_layers:
                    dec_out = layer(dec_out, single_enc, tgt_pad, tgt_causal, single_mask)
                dec_out = model.decoder_final_ln(dec_out)
                
                logits = F.linear(dec_out, model.embedding.weight, model.output_bias)
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                
                if top_k and top_k > 0:
                    top_val, top_idx = torch.topk(log_probs, min(top_k, log_probs.size(-1)))
                    mask = torch.full_like(log_probs, float("-inf"))
                    mask.scatter_(1, top_idx, top_val)
                    log_probs = mask
                
                top_vals, top_idx = torch.topk(log_probs.squeeze(0), beam_size)
                for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
                    new_beams.append(BeamSearchHypothesis(
                        tokens=beam.tokens + [idx],
                        log_prob=beam.log_prob + val
                    ))
            
            beams = sorted(new_beams, key=lambda h: h.log_prob, reverse=True)[:beam_size]
            if not beams:
                break
        
        finished.extend(beams)
        best = max(finished, key=lambda h: h.log_prob / (len(h.tokens) ** length_penalty))
        tokens = best.tokens[1:]
        if eos in tokens:
            tokens = tokens[: tokens.index(eos)]
        decoded.append(sp_model.decode(tokens))
    
    return decoded


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    sp_model: spm.SentencePieceProcessor,
    config: RopeConfig,
    calculate_bleu: bool = True,
    max_bleu_samples: int = 500,
) -> tuple:
    """
    Evaluate model on validation set.
    
    Args:
        model: Transformer model
        dataloader: Validation dataloader
        criterion: Loss criterion
        sp_model: SentencePiece tokenizer
        config: Configuration
        calculate_bleu: Whether to calculate BLEU score
        max_bleu_samples: Maximum samples for BLEU calculation
        
    Returns:
        Tuple of (average_loss, bleu_score)
    """
    model.eval()
    total_loss = 0.0
    all_predictions, all_references = [], []
    bleu_count = 0

    for i, (src_batch, tgt_batch) in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
        src_batch = src_batch.to(config.device)
        tgt_batch = tgt_batch.to(config.device)

        # Calculate loss
        logits = model(src_batch, tgt_batch)
        targets = tgt_batch[:, 1:]
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        total_loss += loss.item()

        # Calculate BLEU on subset for speed
        if calculate_bleu and bleu_count < max_bleu_samples:
            preds = greedy_decode(model, src_batch, sp_model, config, max_len=config.max_len)

            eos = sp_model.piece_to_id(config.eos_token)
            for tgt_seq in tgt_batch:
                ref = tgt_seq[1:].cpu().tolist()
                if eos in ref:
                    ref = ref[: ref.index(eos)]
                all_references.append(sp_model.decode(ref))

            all_predictions.extend(preds)
            bleu_count += len(preds)

            if bleu_count >= max_bleu_samples:
                break

    avg_loss = total_loss / len(dataloader)
    bleu_score = 0.0

    if calculate_bleu and sacrebleu is not None and all_predictions:
        try:
            bleu = sacrebleu.corpus_bleu(all_predictions, [all_references], force=True)
            bleu_score = bleu.score
        except Exception as err:
            print(f"BLEU calculation failed: {err}")

    return avg_loss, bleu_score
