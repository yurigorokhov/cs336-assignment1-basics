#!/usr/bin/env python3

import argparse
import cProfile
import pstats
import sys
import time
import logging

from cs336_basics.tokenizer import run_train_bpe, save_tokenizer


def with_timing(handler_func):
    """Decorator that adds timing to command handlers."""

    def wrapper(args):
        start_time = time.time()
        try:
            result = handler_func(args)
            elapsed_time = time.time() - start_time
            print(f"Command completed in {elapsed_time:.2f} seconds")
            return result
        except Exception:
            elapsed_time = time.time() - start_time
            print(f"Command failed after {elapsed_time:.2f} seconds")
            raise

    return wrapper


def with_profiling(handler_func):
    """Decorator that adds cProfile profiling to command handlers."""

    def wrapper(args):
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            result = handler_func(args)
            profiler.disable()

            # Save profile to file
            profile_filename = f"profile_{args.command}.prof"
            profiler.dump_stats(profile_filename)
            print(f"Profile saved to {profile_filename}")

            # Print basic stats
            stats = pstats.Stats(profiler)
            stats.sort_stats("cumulative")
            print("\nTop 10 functions by cumulative time:")
            stats.print_stats(10)

            return result
        except Exception:
            profiler.disable()
            raise

    return wrapper


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="CLI tool skeleton")
    parser.add_argument("--profile", action="store_true", help="Enable profiling with cProfile")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train tokenizer subcommand
    cmd_train = subparsers.add_parser("train-tokenizer", help="Train a BPE tokenizer")
    cmd_train.add_argument("input", help="Path to training corpus file")
    cmd_train.add_argument("--output", "-o", required=True, help="Path to save the trained tokenizer")
    cmd_train.add_argument("--vocab-size", type=int, default=1000, help="Size of the vocabulary (default: 1000)")
    cmd_train.add_argument("--special-tokens", action="append", help="Special tokens (can be specified multiple times)")
    cmd_train.add_argument(
        "--pretokenizer-num-chunks",
        type=int,
        default=None,
        help="Number of chunks to pre-tokenize with (default: #cpu)",
    )
    cmd_train.add_argument("--verbose", "-v", action="store_true", help="Show training progress")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    try:
        # Command handlers mapping
        base_handler = handle_train_tokenizer

        # Apply decorators based on flags
        handler = with_timing(base_handler)
        if args.profile:
            handler = with_profiling(handler)

        handlers = {
            "train-tokenizer": handler,
        }

        handler = handlers.get(args.command)
        if handler:
            return handler(args)
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            return 1
    except Exception as e:
        if args.verbose:
            raise
        print(f"Error: {e}", file=sys.stderr)
        return 1


def handle_train_tokenizer(args) -> int:
    special_tokens = args.special_tokens or ["<|endoftext|>"]

    progress_callback = None
    if args.verbose:

        def progress_callback(iter):
            logging.info(f"Iteration={iter}")

    vocab, merges = run_train_bpe(
        args.input,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        progress_callback=progress_callback,
        pretokenizer_num_chunks=args.pretokenizer_num_chunks,
    )

    save_tokenizer(args.output, vocab, merges)
    print(f"Tokenizer saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
