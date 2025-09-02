#!/usr/bin/env python3

import argparse
import sys
import time

from cs336_basics.tokenizer import run_train_bpe, save_tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="CLI tool skeleton")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train tokenizer subcommand
    cmd_train = subparsers.add_parser("train-tokenizer", help="Train a BPE tokenizer")
    cmd_train.add_argument("input", help="Path to training corpus file")
    cmd_train.add_argument("--output", "-o", required=True, help="Path to save the trained tokenizer")
    cmd_train.add_argument("--vocab-size", type=int, default=1000, help="Size of the vocabulary (default: 1000)")
    cmd_train.add_argument("--special-tokens", action="append", help="Special tokens (can be specified multiple times)")
    cmd_train.add_argument("--verbose-progress", action="store_true", help="Show training progress")

    # Example subcommand
    cmd_example = subparsers.add_parser("example", help="Example command")
    cmd_example.add_argument("input", help="Input parameter")
    cmd_example.add_argument("--output", "-o", help="Output file")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    try:
        if args.command == "train-tokenizer":
            return handle_train_tokenizer(args)
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            return 1
    except Exception as e:
        if args.verbose:
            raise
        print(f"Error: {e}", file=sys.stderr)
        return 1


def handle_train_tokenizer(args) -> int:
    start_time = time.time()

    special_tokens = args.special_tokens or ["<|endoftext|>"]

    progress_callback = None
    if args.verbose_progress:
        def progress_callback(iter):
            return print(f"Iteration={iter}")

    vocab, merges = run_train_bpe(
        args.input,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        progress_callback=progress_callback
    )

    save_tokenizer(args.output, vocab, merges)

    elapsed_time = time.time() - start_time
    print(f"Tokenizer saved to {args.output}")
    print(f"Training completed in {elapsed_time:.2f} seconds")

    return 0


if __name__ == "__main__":
    sys.exit(main())