"""Data processing utilities for Vietnamese-Chinese translation."""

import os
import sentencepiece as spm
from typing import List, Tuple
import numpy as np


class DataProcessor:
    """Handles data loading, filtering, and tokenization."""

    def __init__(self, spm_prefix: str, max_tokens: int = 32):
        """
        Initialize data processor.
        
        Args:
            spm_prefix: Path prefix for SentencePiece model
            max_tokens: Maximum token length for filtering
        """
        self.spm_prefix = spm_prefix
        self.max_tokens = max_tokens
        self.sp = None

    def load_tokenizer(self) -> spm.SentencePieceProcessor:
        """Load SentencePiece tokenizer."""
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(f"{self.spm_prefix}.model")
        return self.sp

    def train_tokenizer(
        self,
        src_file: str,
        tgt_file: str,
        vocab_size: int = 8000,
        character_coverage: float = 1.0,
        user_symbols: List[str] = None,
    ) -> None:
        """
        Train SentencePiece tokenizer on parallel corpus.
        
        Args:
            src_file: Path to source language file
            tgt_file: Path to target language file
            vocab_size: Vocabulary size
            character_coverage: Character coverage for training
            user_symbols: List of special tokens to add
        """
        if user_symbols is None:
            user_symbols = ["<2zh>", "<2vi>"]

        # Create temporary corpus file
        temp_corpus = f"{self.spm_prefix}_temp_corpus.txt"
        with open(temp_corpus, "w", encoding="utf-8") as fout:
            with open(src_file, "r", encoding="utf-8") as f_src, \
                 open(tgt_file, "r", encoding="utf-8") as f_tgt:
                for src_line, tgt_line in zip(f_src, f_tgt):
                    fout.write(src_line)
                    fout.write(tgt_line)

        spm_args = (
            f"--input={temp_corpus} ",
            f"--model_prefix={self.spm_prefix} ",
            f"--vocab_size={vocab_size} ",
            f"--model_type=bpe ",
            f"--character_coverage={character_coverage} ",
            "--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 ",
            f"--user_defined_symbols={','.join(user_symbols)}"
        )
        spm.SentencePieceTrainer.Train(' '.join(spm_args))
        os.remove(temp_corpus)
        self.load_tokenizer()
        print(f"Tokenizer saved to {self.spm_prefix}.model")

    @staticmethod
    def load_lines(path: str) -> List[str]:
        """Load lines from file."""
        with open(path, "r", encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]

    def get_statistics(self, src_lines: List[str], tgt_lines: List[str]) -> dict:
        """Calculate corpus statistics."""
        src_lens = np.array([len(s.split()) for s in src_lines])
        tgt_lens = np.array([len(t.split()) for t in tgt_lines])
        
        return {
            "src": {
                "mean": src_lens.mean(),
                "min": src_lens.min(),
                "max": src_lens.max(),
                "total": len(src_lines)
            },
            "tgt": {
                "mean": tgt_lens.mean(),
                "min": tgt_lens.min(),
                "max": tgt_lens.max(),
                "total": len(tgt_lines)
            }
        }

    def filter_by_token_length(
        self,
        src_lines: List[str],
        tgt_lines: List[str],
        src_prefix: str = "<2vi>",
    ) -> Tuple[List[str], List[str]]:
        """
        Filter sentence pairs by token length.
        
        Args:
            src_lines: Source language sentences
            tgt_lines: Target language sentences
            src_prefix: Language token prefix for source
            
        Returns:
            Tuple of filtered source and target lines
        """
        if self.sp is None:
            self.load_tokenizer()

        filtered_src, filtered_tgt = [], []
        for src, tgt in zip(src_lines, tgt_lines):
            src_with_prefix = f"{src_prefix} {src}"
            src_len = len(self.sp.encode(src_with_prefix, out_type=int))
            tgt_len = len(self.sp.encode(tgt, out_type=int))
            
            if src_len <= self.max_tokens and tgt_len <= self.max_tokens:
                filtered_src.append(src)
                filtered_tgt.append(tgt)

        print(f"Filtered: {len(filtered_src)}/{len(src_lines)} pairs kept "
              f"({100*len(filtered_src)/len(src_lines):.1f}%)")
        return filtered_src, filtered_tgt

    def save_cleaned_data(
        self,
        src_lines: List[str],
        tgt_lines: List[str],
        output_dir: str,
    ) -> Tuple[str, str]:
        """
        Save cleaned data to files.
        
        Args:
            src_lines: Source sentences
            tgt_lines: Target sentences
            output_dir: Output directory
            
        Returns:
            Tuple of output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        src_out = os.path.join(output_dir, f"train_maxlen{self.max_tokens}.zh")
        tgt_out = os.path.join(output_dir, f"train_maxlen{self.max_tokens}.vi")

        with open(src_out, "w", encoding="utf-8") as f:
            f.write("\n".join(src_lines))
        with open(tgt_out, "w", encoding="utf-8") as f:
            f.write("\n".join(tgt_lines))

        return src_out, tgt_out
