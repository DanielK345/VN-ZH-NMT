"""High-level inference interface."""

import os
from typing import List, Optional
import pandas as pd
import sentencepiece as spm
import torch
from tqdm import tqdm

from .model import load_model_from_checkpoint
from .decoder import beam_search_decode, greedy_decode
from .config import InferenceConfig


class Translator:
    """High-level interface for translation inference."""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize translator from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to use (default: auto-detect)
        """
        self.checkpoint_path = checkpoint_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model from {checkpoint_path}...")
        self.model, self.sp_model, self.config = load_model_from_checkpoint(
            checkpoint_path,
            device=self.device
        )
        self.config.device = self.device
        print(f"✓ Model loaded. Using device: {self.device}")
    
    def translate_sentence(
        self,
        text: str,
        src_lang: str = "zh",
        use_beam_search: bool = True,
    ) -> str:
        """
        Translate a single sentence.
        
        Args:
            text: Input sentence
            src_lang: Source language ("zh" for Chinese, "vi" for Vietnamese)
            use_beam_search: Use beam search (True) or greedy decoding (False)
            
        Returns:
            Translated sentence
        """
        # Determine language token
        lang_token = self.config.vi_token if src_lang == "zh" else self.config.zh_token
        
        # Add language token
        text_with_token = f"{lang_token} {text.strip()}"
        
        # Tokenize
        src_ids = self.sp_model.encode(text_with_token, out_type=int)
        src_ids = src_ids[:self.config.max_len]
        src_tensor = torch.tensor(src_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Decode
        if use_beam_search:
            translations = beam_search_decode(
                self.model,
                src_tensor,
                self.sp_model,
                self.config,
            )
        else:
            translations = greedy_decode(
                self.model,
                src_tensor,
                self.sp_model,
                self.config,
            )
        
        return translations[0]
    
    def translate_batch(
        self,
        texts: List[str],
        src_lang: str = "zh",
        use_beam_search: bool = True,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[str]:
        """
        Translate a batch of sentences.
        
        Args:
            texts: List of input sentences
            src_lang: Source language
            use_beam_search: Use beam search or greedy decoding
            batch_size: Number of samples per batch
            show_progress: Show progress bar
            
        Returns:
            List of translated sentences
        """
        translations = []
        lang_token = self.config.vi_token if src_lang == "zh" else self.config.zh_token
        
        iterator = tqdm(
            range(0, len(texts), batch_size),
            desc="Translating",
            disable=not show_progress
        )
        
        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            # Tokenize batch
            batch_ids = []
            for text in batch_texts:
                text_with_token = f"{lang_token} {text.strip()}"
                src_ids = self.sp_model.encode(text_with_token, out_type=int)
                src_ids = src_ids[:self.config.max_len]
                batch_ids.append(src_ids)
            
            # Pad to same length
            max_len = max(len(ids) for ids in batch_ids)
            padded_batch = []
            for ids in batch_ids:
                padded = ids + [0] * (max_len - len(ids))
                padded_batch.append(padded)
            
            src_tensor = torch.tensor(padded_batch, dtype=torch.long, device=self.device)
            
            # Decode
            if use_beam_search:
                batch_translations = beam_search_decode(
                    self.model,
                    src_tensor,
                    self.sp_model,
                    self.config,
                )
            else:
                batch_translations = greedy_decode(
                    self.model,
                    src_tensor,
                    self.sp_model,
                    self.config,
                )
            
            translations.extend(batch_translations)
        
        return translations
    
    def translate_file(
        self,
        input_path: str,
        output_path: str,
        src_lang: str = "zh",
        use_beam_search: bool = True,
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """
        Translate sentences from a file.
        
        Args:
            input_path: Path to input file (one sentence per line)
            output_path: Path to save output CSV
            src_lang: Source language
            use_beam_search: Use beam search or greedy decoding
            batch_size: Batch size for processing
            
        Returns:
            DataFrame with translations
        """
        # Read input file
        with open(input_path, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
        
        print(f"Read {len(sentences)} sentences from {input_path}")
        
        # Translate
        translations = self.translate_batch(
            sentences,
            src_lang=src_lang,
            use_beam_search=use_beam_search,
            batch_size=batch_size,
            show_progress=True,
        )
        
        # Create DataFrame
        if src_lang == "zh":
            df = pd.DataFrame({
                "tieng_trung": sentences,
                "tieng_viet": translations
            })
        else:
            df = pd.DataFrame({
                "tieng_viet": sentences,
                "tieng_trung": translations
            })
        
        # Save to CSV
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"✓ Saved {len(df)} translations to {output_path}")
        
        return df
    
    def set_beam_params(
        self,
        beam_size: int = None,
        top_k: int = None,
        length_penalty: float = None,
    ):
        """Update beam search parameters."""
        if beam_size is not None:
            self.config.beam_size = beam_size
        if top_k is not None:
            self.config.top_k = top_k
        if length_penalty is not None:
            self.config.length_penalty = length_penalty
