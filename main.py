#!/usr/bin/env python3

import argparse
import cProfile
import os
import pstats
import sys
import time
import logging
import yaml

from cs336_basics.tokenizer import run_train_bpe, save_tokenizer, load_tokenizer, Tokenizer
from cs336_basics.train import training_loop
from cs336_basics.config import instantiate, set_nested
from cs336_basics.generate import LLMServing


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
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="CLI tool skeleton")
    parser.add_argument("--profile", action="store_true", help="Enable profiling with cProfile")
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train tokenizer subcommand
    cmd_train = subparsers.add_parser("train-tokenizer", help="Train a BPE tokenizer")
    cmd_train.add_argument("input", help="Path to training corpus file")
    cmd_train.add_argument(
        "--output", "-o", required=True, help="Path to save the trained tokenizer"
    )
    cmd_train.add_argument(
        "--vocab-size", type=int, default=1000, help="Size of the vocabulary (default: 1000)"
    )
    cmd_train.add_argument(
        "--special-tokens", action="append", help="Special tokens (can be specified multiple times)"
    )
    cmd_train.add_argument(
        "--pretokenizer-num-chunks",
        type=int,
        default=None,
        help="Number of chunks to pre-tokenize with (default: #cpu)",
    )
    cmd_train.add_argument("--verbose", "-v", action="store_true", help="Show training progress")

    # Tokenize dataset subcommand
    cmd_tokenize = subparsers.add_parser(
        "tokenize-dataset", help="Tokenize a dataset using a trained tokenizer"
    )
    cmd_tokenize.add_argument("input", help="Path to input text file to tokenize")
    cmd_tokenize.add_argument(
        "--tokenizer", "-t", required=True, help="Path to tokenizer msgpack file"
    )
    cmd_tokenize.add_argument("--output", "-o", required=True, help="Path to save tokenized output")
    cmd_tokenize.add_argument(
        "--special-tokens", action="append", help="Special tokens (can be specified multiple times)"
    )
    cmd_tokenize.add_argument(
        "--chunks", type=int, default=1000, help="Number of chunks to split the dataset into"
    )

    # Train model subcommand
    cmd_train = subparsers.add_parser("train", help="Train a model using a config file")
    cmd_train.add_argument("config_file", help="Path to YAML config file")
    cmd_train.add_argument(
        "--entity", default="yurigorokhov-personal", help="WandB entity (team name)"
    )
    cmd_train.add_argument(
        "--project", default="transformer-assignment-1", help="WandB project name"
    )
    cmd_train.add_argument("--offline", action="store_true", help="Run WandB in offline mode")
    cmd_train.add_argument(
        "--disable-wandb", action="store_true", help="Disable WandB logging entirely"
    )
    cmd_train.add_argument(
        "--random-checkpoint-dir", action="store_true", help="Run WandB in offline mode"
    )

    # Chat subcommand
    cmd_chat = subparsers.add_parser("chat", help="Interactive chat with a trained model")
    cmd_chat.add_argument("config_file", help="Path to YAML config file")
    cmd_chat.add_argument("--checkpoint", "-c", required=True, help="Path to model checkpoint")
    cmd_chat.add_argument(
        "--prompt",
        "-p",
        default="Once upon a time, in a warm and sunny place",
        help="Initial prompt for generation",
    )
    cmd_chat.add_argument(
        "--max-tokens", type=int, default=256, help="Maximum tokens to generate (default: 256)"
    )
    cmd_chat.add_argument(
        "--temperature", type=float, default=None, help="Sampling temperature (default: None)"
    )
    cmd_chat.add_argument(
        "--top-p", type=float, default=None, help="Top-p nucleus sampling threshold (default: None)"
    )

    # Dataset info subcommand
    cmd_dataset_info = subparsers.add_parser(
        "dataset-info", help="Show statistics about a tokenized dataset"
    )
    cmd_dataset_info.add_argument(
        "dataset_path", help="Path to dataset directory containing ArrayRecord files"
    )
    cmd_dataset_info.add_argument(
        "--tokenizer", "-t", help="Path to tokenizer file (optional, for document counting)"
    )

    # Validate dataloader subcommand
    cmd_validate = subparsers.add_parser(
        "validate-dataloader",
        help="Validate data loader configuration by inspecting training examples",
    )
    cmd_validate.add_argument("config_file", help="Path to YAML config file")
    cmd_validate.add_argument(
        "--loader",
        type=str,
        default="data_loader",
        choices=["data_loader", "validation_data_loader"],
        help="Which data loader to validate (default: data_loader)",
    )
    cmd_validate.add_argument(
        "--show-examples",
        type=int,
        default=0,
        help="Number of examples to show per batch (default: 0)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    try:
        # Command handlers mapping
        if args.command == "train-tokenizer":
            base_handler = handle_train_tokenizer
        elif args.command == "tokenize-dataset":
            base_handler = handle_tokenize_dataset
        elif args.command == "train":
            base_handler = handle_train
        elif args.command == "chat":
            base_handler = handle_chat
        elif args.command == "dataset-info":
            base_handler = handle_dataset_info
        elif args.command == "validate-dataloader":
            base_handler = handle_validate_dataloader
        else:
            base_handler = None

        # Apply decorators based on flags
        if base_handler:
            handler = with_timing(base_handler)
            if args.profile:
                handler = with_profiling(handler)

        handlers = {
            "train-tokenizer": handler if args.command == "train-tokenizer" else None,
            "tokenize-dataset": handler if args.command == "tokenize-dataset" else None,
            "train": handler if args.command == "train" else None,
            "chat": handler if args.command == "chat" else None,
            "dataset-info": handler if args.command == "dataset-info" else None,
            "validate-dataloader": handler if args.command == "validate-dataloader" else None,
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


def handle_tokenize_dataset(args) -> int:
    vocab, merges = load_tokenizer(args.tokenizer)
    special_tokens = args.special_tokens or ["<|endoftext|>"]
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    tokenizer.tokenize_dataset_to_array_record(args.input, args.output, args.chunks)
    print(f"Tokenized dataset saved to {args.output}")
    return 0


def handle_train(args) -> int:
    import wandb

    # Load config
    with open(args.config_file) as file:
        config = yaml.safe_load(file)

    # Initialize wandb
    wandb_mode = "disabled" if args.disable_wandb else ("offline" if args.offline else "online")
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        config=config,
        mode=wandb_mode,
    )

    # apply wandb parameter overrides
    for k, v in wandb.config.items():
        set_nested(config, k, v)

    # Update checkpoint directory to include wandb run info for tracking
    # This ensures each sweep run has its own checkpoint directory
    if wandb_mode != "disabled" and config.get("checkpoint_dir") is None:
        base_checkpoint_dir = "./scratch/checkpoints/"
        config["checkpoint_dir"] = os.path.join(base_checkpoint_dir, f"{run.id}_{run.name}")

    # Run training
    training_loop(**instantiate(config), run=run)

    print("Training completed!")
    return 0


def handle_chat(args) -> int:
    # Load config (only tokenizer and model sections)
    with open(args.config_file) as file:
        config = {k: v for k, v in yaml.safe_load(file).items() if k in ["tokenizer", "model"]}

    # Instantiate model and tokenizer
    entities = instantiate(config)

    # Initialize LLM serving
    generator = LLMServing(
        entities["model"],
        entities["tokenizer"],
        checkpoint=args.checkpoint,
    )

    # Generate text
    generation_kwargs = dict(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    for chunk in generator.generate(args.prompt, **generation_kwargs):
        print(chunk, end="", flush=True)

    print()
    return 0


def handle_dataset_info(args) -> int:
    from glob import glob
    import grain.python as grain
    import numpy as np

    # Find all ArrayRecord files
    pattern = os.path.join(args.dataset_path, "*.arrayrecord")
    files = sorted(glob(pattern))

    if not files:
        print(f"No ArrayRecord files found in {args.dataset_path}")
        return 1

    print(f"Dataset: {args.dataset_path}")
    print(f"Number of shards: {len(files)}")
    print()

    # Load dataset
    ds = grain.MapDataset.source(grain.ArrayRecordDataSource(files))

    # Count tokens per shard
    total_tokens = 0
    shard_tokens = []

    for elem in ds:
        tokens = np.frombuffer(elem, dtype=np.uint16)
        num_tokens = len(tokens)
        shard_tokens.append(num_tokens)
        total_tokens += num_tokens

    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens per shard: {total_tokens / len(files):,.0f}")
    print(f"Min tokens in a shard: {min(shard_tokens):,}")
    print(f"Max tokens in a shard: {max(shard_tokens):,}")

    # Count documents if tokenizer provided
    if args.tokenizer:
        vocab, _ = load_tokenizer(args.tokenizer)

        # Find <endoftext> token ID
        endoftext_id = None
        for token_id, token_bytes in vocab.items():
            if token_bytes.decode("utf-8", errors="replace") == "<|endoftext|>":
                endoftext_id = token_id
                break

        if endoftext_id is not None:
            total_docs = 0
            for elem in ds:
                tokens = np.frombuffer(elem, dtype=np.uint16)
                doc_count = np.sum(tokens == endoftext_id)
                total_docs += doc_count

            print()
            print(f"Number of <|endoftext|> tokens: {total_docs:,}")
            if total_docs > 0:
                print(f"Average tokens per document: {total_tokens / total_docs:,.1f}")
        else:
            print()
            print("Warning: Could not find <|endoftext|> token in vocabulary")

    return 0


def handle_validate_dataloader(args) -> int:
    import numpy as np

    # Load config
    with open(args.config_file) as file:
        config = yaml.safe_load(file)

    # Extract data loader and tokenizer configs
    loader_key = args.loader
    dataloader_config = config.get(loader_key)
    tokenizer_config = config.get("tokenizer")

    if not dataloader_config:
        print(f"Error: No '{loader_key}' section found in config", file=sys.stderr)
        return 1

    if not tokenizer_config:
        print("Error: No 'tokenizer' section found in config", file=sys.stderr)
        return 1

    # Instantiate tokenizer and data loader
    entities = instantiate(config)
    tokenizer = entities["tokenizer"]
    data_loader_fn = entities[loader_key]
    data_loader = data_loader_fn()

    print("=" * 80)
    print("DATA LOADER VALIDATION")
    print("=" * 80)
    print(f"Config file: {args.config_file}")
    print(f"Loader: {loader_key}")
    print(f"Dataset path: {dataloader_config.get('dataset_path')}")
    print(f"Sequence length: {dataloader_config.get('sequence_length')}")
    print(f"Batch size: {dataloader_config.get('batch_size')}")
    print(f"Samples per file: {dataloader_config.get('samples_per_file')}")
    print(f"Num epochs: {dataloader_config.get('num_epochs')}")
    print()

    # Track unique examples to detect duplicates
    seen_examples = set()
    total_examples = 0
    duplicate_count = 0
    total_tokens = 0

    # Inspect batches
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        # Convert to numpy for easier inspection
        inputs_np = inputs.numpy() if hasattr(inputs, "numpy") else inputs
        targets_np = targets.numpy() if hasattr(targets, "numpy") else targets

        # Count total tokens
        total_tokens += inputs_np.size

        if args.show_examples > 0:
            print(f"Batch {batch_idx + 1}")
            print(f"  Shape: inputs={inputs_np.shape}, targets={targets_np.shape}")

        # Check all examples for duplicates
        num_examples_to_check = inputs_np.shape[0]
        num_examples_to_show = min(args.show_examples, inputs_np.shape[0])

        for i in range(num_examples_to_check):
            input_seq = inputs_np[i]
            target_seq = targets_np[i]

            # Create a hash of the example
            example_hash = hash(input_seq.tobytes())
            total_examples += 1

            if example_hash in seen_examples:
                duplicate_count += 1
                if i < num_examples_to_show:
                    print(f"  Example {i + 1}: ⚠️  DUPLICATE DETECTED")
            else:
                seen_examples.add(example_hash)

            # Only show details for the first few examples
            if i < num_examples_to_show:
                # Show the example
                print(f"  Example {i + 1}:")

                # Decode and show first few tokens
                input_tokens = [tokenizer.vocab[int(idx)] for idx in input_seq[:10]]
                target_tokens = [tokenizer.vocab[int(idx)] for idx in target_seq[:10]]

                input_text = "".join(
                    [token.decode("utf-8", errors="replace") for token in input_tokens]
                )
                target_text = "".join(
                    [token.decode("utf-8", errors="replace") for token in target_tokens]
                )

                print(f"    Input (first 10):  {input_text[:50]!r}")
                print(f"    Target (first 10): {target_text[:50]!r}")

                # Verify shift property: targets should be inputs shifted by 1
                if not np.array_equal(input_seq[1:], target_seq[:-1]):
                    print("    ❌ ERROR: Target is not properly shifted from input!")
                else:
                    print("    ✓ Target is correctly shifted from input")

                print()

        if args.show_examples > 0:
            print()

    # Summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total batches: {batch_idx + 1}")
    print(f"Total examples inspected: {total_examples}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Unique examples: {len(seen_examples)}")
    print(f"Duplicate examples: {duplicate_count}")

    if duplicate_count > 0:
        print(f"⚠️  WARNING: Found {duplicate_count} duplicate examples!")
        print("   This suggests the data loader may be repeating the same data.")
    else:
        print("✓ No duplicates detected in sampled batches")

    if len(seen_examples) == 1:
        print("❌ CRITICAL: All examples are identical!")
        print("   Check the data loader implementation for bugs.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
