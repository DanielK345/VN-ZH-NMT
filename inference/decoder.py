"""Decoding strategies for inference."""

from dataclasses import dataclass
from typing import List
import torch
import torch.nn.functional as F
import sentencepiece as spm

from .config import InferenceConfig


@dataclass
class BeamSearchHypothesis:
    """Hypothesis for beam search."""
    tokens: List[int]
    log_prob: float

    def __lt__(self, other):
        return self.log_prob < other.log_prob


@torch.no_grad()
def greedy_decode(
    model,
    src_ids: torch.Tensor,
    sp_model: spm.SentencePieceProcessor,
    config: InferenceConfig,
    max_len: int = None,
) -> List[str]:
    """
    Fast greedy decoding.
    
    Args:
        model: TransformerInference model
        src_ids: Source token IDs [batch_size, src_len]
        sp_model: SentencePiece tokenizer
        config: Inference configuration
        max_len: Maximum decoding length
        
    Returns:
        List of decoded strings
    """
    model.eval()
    batch_size = src_ids.size(0)
    device = src_ids.device
    max_len = max_len or config.max_len
    
    bos_id = sp_model.piece_to_id(config.bos_token)
    eos_id = sp_model.piece_to_id(config.eos_token)
    pad_id = sp_model.piece_to_id(config.pad_token)
    
    # Encode source
    src_pad_mask = (src_ids == pad_id)
    enc_out = model.encode(src_ids, src_pad_mask)
    
    # Initialize decoder input with BOS
    tgt_ids = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    
    # Decode step by step
    for _ in range(max_len):
        tgt_pad_mask = (tgt_ids == pad_id)
        tgt_len = tgt_ids.size(1)
        tgt_causal = torch.triu(
            torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device),
            diagonal=1
        )
        
        dec_out = model.decode(tgt_ids, enc_out, tgt_pad_mask, tgt_causal, src_pad_mask)
        logits = model.project(dec_out)
        next_tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tgt_ids = torch.cat([tgt_ids, next_tokens], dim=1)
        
        # Stop if all sequences have EOS
        if (next_tokens.squeeze(-1) == eos_id).all():
            break
    
    # Decode to text
    decoded = []
    for seq in tgt_ids:
        tokens = seq[1:].cpu().tolist()  # Skip BOS
        if eos_id in tokens:
            tokens = tokens[:tokens.index(eos_id)]
        decoded.append(sp_model.decode(tokens))
    
    return decoded


@torch.no_grad()
def beam_search_decode(
    model,
    src_ids: torch.Tensor,
    sp_model: spm.SentencePieceProcessor,
    config: InferenceConfig,
    beam_size: int = None,
    top_k: int = None,
    max_len: int = None,
    length_penalty: float = None,
) -> List[str]:
    """
    Beam search decoding with length penalty and top-k filtering.
    
    Args:
        model: TransformerInference model
        src_ids: Source token IDs [batch_size, src_len]
        sp_model: SentencePiece tokenizer
        config: Inference configuration
        beam_size: Beam width (default from config)
        top_k: Top-k filtering (default from config)
        max_len: Maximum decoding length
        length_penalty: Length penalty coefficient
        
    Returns:
        List of decoded strings
    """
    beam_size = beam_size or config.beam_size
    top_k = top_k or config.top_k
    max_len = max_len or config.max_len
    length_penalty = length_penalty or config.length_penalty
    
    model.eval()
    batch_size = src_ids.size(0)
    device = src_ids.device
    
    bos_id = sp_model.piece_to_id(config.bos_token)
    eos_id = sp_model.piece_to_id(config.eos_token)
    pad_id = sp_model.piece_to_id(config.pad_token)
    
    # Encode source
    src_pad_mask = (src_ids == pad_id)
    enc_out = model.encode(src_ids, src_pad_mask)
    
    decoded = []
    
    # Process each batch element separately
    for b in range(batch_size):
        beams = [BeamSearchHypothesis(tokens=[bos_id], log_prob=0.0)]
        finished = []
        
        single_enc = enc_out[b : b + 1]
        single_mask = src_pad_mask[b : b + 1]
        
        # Beam search decoding
        for _ in range(max_len):
            if not beams:
                break
            
            proposals = []
            
            for hyp in beams:
                # Stop if EOS reached
                if hyp.tokens[-1] == eos_id:
                    finished.append(hyp)
                    continue
                
                # Decode next token
                tgt_ids = torch.tensor(hyp.tokens, dtype=torch.long, device=device).unsqueeze(0)
                tgt_pad_mask = (tgt_ids == pad_id)
                tgt_len = tgt_ids.size(1)
                tgt_causal = torch.triu(
                    torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device),
                    diagonal=1
                )
                
                dec_out = model.decode(tgt_ids, single_enc, tgt_pad_mask, tgt_causal, single_mask)
                logits = model.project(dec_out)
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                
                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    top_vals, top_idx = torch.topk(log_probs, min(top_k, log_probs.size(-1)))
                    mask = torch.full_like(log_probs, float('-inf'))
                    mask.scatter_(1, top_idx, top_vals)
                    log_probs = mask
                
                # Get top beam_size candidates
                top_vals, top_idx = torch.topk(log_probs.squeeze(0), beam_size)
                
                for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
                    proposals.append(BeamSearchHypothesis(
                        tokens=hyp.tokens + [idx],
                        log_prob=hyp.log_prob + val
                    ))
            
            # Select top beam_size proposals
            beams = sorted(proposals, key=lambda h: h.log_prob, reverse=True)[:beam_size]
        
        # Select best hypothesis
        finished.extend(beams)
        
        if finished:
            best = max(
                finished,
                key=lambda h: h.log_prob / (len(h.tokens) ** length_penalty)
            )
        else:
            best = BeamSearchHypothesis(tokens=[bos_id, eos_id], log_prob=0.0)
        
        # Decode tokens to text
        tokens = best.tokens[1:]  # Skip BOS
        if eos_id in tokens:
            tokens = tokens[:tokens.index(eos_id)]
        decoded.append(sp_model.decode(tokens))
    
    return decoded
