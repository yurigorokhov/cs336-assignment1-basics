from collections import Counter
import functools
import os
import regex as re
import multiprocessing
from typing import BinaryIO


PRETOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
DEFAULT_PARALLELISM = multiprocessing.cpu_count()


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    vocab: dict[int, bytes] = {i: chr(i).encode("utf-8") for i in range(256)}
    next_free_token = 256

    # assign a token for each special token
    for s_token in special_tokens:
        vocab[next_free_token] = s_token.encode("utf-8")
        next_free_token += 1

    merges: list[tuple[bytes, bytes]] = []
    return vocab, merges


def _pre_tokenize_from_file_byte_range(file_path: str | os.PathLike, boundary: tuple[int, int]) -> Counter[bytes]:
    start, end = boundary
    pre_token_counts: Counter[bytes] = Counter()
    with open(file_path, "rb") as f:
        f.seek(start)
        raw_text = f.read(end - start).decode("utf-8", errors="ignore")

        # pre-tokenize
        for m in re.finditer(PRETOKENIZER_PATTERN, raw_text):
            pre_token = m.group(0).encode("utf-8")
            pre_token_counts[pre_token.decode("utf-8")] += 1 # type: ignore
    return pre_token_counts


def _pre_tokenize_from_file(file_path: str | os.PathLike, split_special_token: bytes, parallelism: int = DEFAULT_PARALLELISM) -> Counter[bytes]:

    # split file boundaries for parallelism
    with open(file_path, "rb") as f:
        boundaries = _find_chunk_boundaries(f, parallelism, split_special_token)

    # count pre-tokens in parallel
    pre_token_counts: Counter[bytes] = Counter()
    with multiprocessing.Pool(parallelism) as pool:
        for result in pool.map(
            functools.partial(_pre_tokenize_from_file_byte_range, file_path),
            zip(boundaries[:-1], boundaries[1:])
        ):
            pre_token_counts += result
    return pre_token_counts


def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
