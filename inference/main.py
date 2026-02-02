"""Main inference script."""

import os
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .inference import Translator


def translate_split(
    checkpoint_path: str,
    split_name: str,
    input_path: str,
    output_path: str,
    src_lang: str = "zh",
    use_beam_search: bool = True,
    batch_size: int = 32,
) -> dict:
    """Translate a single split."""
    translator = Translator(checkpoint_path)
    df = translator.translate_file(
        input_path,
        output_path,
        src_lang=src_lang,
        use_beam_search=use_beam_search,
        batch_size=batch_size,
    )
    return {
        "split": split_name,
        "output_path": output_path,
        "count": len(df),
    }


def main():
    parser = argparse.ArgumentParser(description="Translate using trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input file (one sentence per line)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save output CSV"
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        default="zh",
        choices=["zh", "vi"],
        help="Source language (default: zh)"
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=3,
        help="Beam size for beam search (default: 3)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top-k for filtering (default: 5)"
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=0.6,
        help="Length penalty for beam search (default: 0.6)"
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding instead of beam search"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    # Initialize translator
    translator = Translator(args.checkpoint)
    translator.set_beam_params(
        beam_size=args.beam_size,
        top_k=args.top_k,
        length_penalty=args.length_penalty,
    )
    
    # Translate
    df = translator.translate_file(
        args.input,
        args.output,
        src_lang=args.src_lang,
        use_beam_search=not args.greedy,
        batch_size=args.batch_size,
    )
    
    print(f"\n✓ Translation complete!")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Samples: {len(df)}")
    print(f"  Decoding: {'Beam Search' if not args.greedy else 'Greedy'}")


if __name__ == "__main__":
    main()
