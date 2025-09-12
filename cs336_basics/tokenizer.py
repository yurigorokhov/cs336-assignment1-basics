from collections import Counter
import functools
import itertools
import logging
import msgpack
import os
import regex as re
import multiprocessing
from typing import BinaryIO
from collections.abc import Iterable, Iterator


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

    progress_callback = kwargs.get("progress_callback")
    pretokenizer_num_chunks = kwargs.get("pretokenizer_num_chunks")

    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = dict()
    next_free_token = 0

    # assign a token for each special token
    special_token_bytes = [s_token.encode("utf-8") for s_token in special_tokens]
    for s_token in special_token_bytes:
        vocab[next_free_token] = s_token
        next_free_token += 1

    # initialize vocab with 256 possible byte values
    for i in range(256):
        vocab[next_free_token] = chr(i).encode("latin-1")
        next_free_token += 1

    # pre-tokenize file
    logging.info("starting pre-tokenizing")
    pre_token_counts = _pre_tokenize_from_file(input_path, special_token_bytes, num_chunks=pretokenizer_num_chunks)
    logging.info(f"finished pre-tokenizing, {len(pre_token_counts)} pre-tokens found.")

    byte_pair_counter: Counter[tuple[bytes, bytes]] = Counter()
    for pre_token, count in pre_token_counts.items():
        # iterate over byte pairs
        byte_pairs = zip(pre_token[:-1], pre_token[1:])
        for pair in byte_pairs:
            byte_pair_counter[pair] += count

    iteration = 0
    while len(vocab) < vocab_size:
        iteration += 1
        if progress_callback and iteration % 1000 == 0:
            progress_callback(iteration)

        # find most common byte pair to merge next
        most_common_pair_count = byte_pair_counter.most_common(1)[0][1]
        tied_keys = [k for k, v in byte_pair_counter.items() if v == most_common_pair_count]
        most_common = max(tied_keys)

        # add new byte-pair to vocab
        joined_pair = b"".join(most_common)
        vocab[next_free_token] = joined_pair
        next_free_token += 1

        # byte pair stops existing after merge, don't count it anymore
        byte_pair_counter.pop(most_common)

        # merge pre-tokens
        merges.append(most_common)

        # merge pre-tokens
        pre_token_replacements = _merge_pretokens(pre_token_counts, most_common, joined_pair, byte_pair_counter)

        # replace pre-tokens that have updates
        for replace, replace_with in pre_token_replacements:
            # pre-tokens of length 1 will not yield any future pairs, we can discard
            if replace_with is None or len(replace_with) == 1:
                pre_token_counts.pop(replace)
            else:
                pre_token_counts[replace_with] = pre_token_counts.pop(replace)

    return vocab, merges


def save_tokenizer(file_path: os.PathLike, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]):
    d = msgpack.packb(
        {
            "vocab": {k: v for k, v in vocab.items()},
            "merges": merges,
        }
    )
    with open(file_path, "wb") as fp:
        fp.write(d)


def load_tokenizer(file_path: os.PathLike) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(file_path, "rb") as fp:
        d = msgpack.load(fp, strict_map_key=False)
    return d["vocab"], d["merges"]


def _merge_pretokens(
    pre_token_counts: Counter[tuple[bytes]],
    merge_pair: tuple[bytes, bytes],
    replace_with: bytes,
    byte_pair_counter: Counter[tuple[bytes, bytes]],
) -> list[tuple[tuple[bytes], tuple[bytes]]]:
    pre_token_replacements: list[tuple[tuple[bytes], tuple[bytes]]] = []

    merge_pair_left, merge_pair_right = merge_pair

    for pre_token, count in pre_token_counts.items():
        # if pre-token contains the sequence (most_common), merge and adjust counts of surrounding tokens
        byte_pairs = zip(
            # previous
            (None,) + pre_token[:-1],
            pre_token[:-1],
            pre_token[1:],
            # next
            pre_token[2:] + (None,),
        )
        matches = 0
        skip = False
        for prev, left, right, next in byte_pairs:
            if skip:
                skip = False
                continue
            if left == merge_pair_left and right == merge_pair_right:
                skip = True
                matches += 1
                # update counts with surrounding byte pairs
                if prev:
                    byte_pair_counter[(prev, left)] -= count
                    byte_pair_counter[(prev, replace_with)] += count
                if next:
                    byte_pair_counter[(right, next)] -= count
                    byte_pair_counter[(replace_with, next)] += count

        # OPTIMIZATION: only if match is found, go back and perform token merges
        if matches:
            # OPTIMIZATION: if we are going to collapse into a single pre_token, just remove it!
            if len(pre_token) == 2:
                pre_token_replacements.append((pre_token, None))
            else:
                updated_pre_token: list[bytes] = []
                prev_token_merged = False
                for left, right in itertools.pairwise(pre_token):
                    if not prev_token_merged and (left == merge_pair_left and right == merge_pair_right):
                        updated_pre_token.append(replace_with)
                        prev_token_merged = True
                    else:
                        if prev_token_merged:
                            prev_token_merged = False
                        else:
                            updated_pre_token.append(left)
                if not prev_token_merged:
                    updated_pre_token.append(right)

                pre_token_replacements.append((pre_token, tuple(updated_pre_token)))  # type: ignore

    return pre_token_replacements


def _pre_tokenize_from_file_byte_range(
    file_path: str | os.PathLike, boundary: tuple[int, int], split_special_tokens: list[bytes]
) -> Counter[tuple[bytes]]:
    start, end = boundary
    pre_token_counts: Counter[tuple[bytes]] = Counter()
    with open(file_path, "rb") as f:
        f.seek(start)
        raw_text = f.read(end - start).decode("utf-8", errors="ignore")

        # pre-tokenize while splitting on special tokens so that we don't accidentally go accross them with the regex
        special_token_regex = "|".join([re.escape(s.decode("utf-8"), special_only=True) for s in split_special_tokens])
        for paragraph in re.splititer(special_token_regex, raw_text):
            for m in re.finditer(PRETOKENIZER_PATTERN, paragraph):
                pre_token: tuple[bytes] = tuple(x.to_bytes() for x in m.group(0).encode("utf-8"))  # type: ignore

                # pre-tokens of len 1 will not yield any byte-pairs
                if len(pre_token) > 1:
                    pre_token_counts[pre_token] += 1
    return pre_token_counts


def _pre_tokenize_from_file(
    file_path: str | os.PathLike,
    split_special_tokens: list[bytes],
    parallelism: int = DEFAULT_PARALLELISM,
    num_chunks: int | None = None,
) -> Counter[tuple[bytes]]:
    # split file boundaries for parallelism
    with open(file_path, "rb") as f:
        boundaries = _find_chunk_boundaries(f, num_chunks or parallelism, split_special_tokens)
    logging.info(f"pre-tokenizing {len(boundaries[:-1])} chunks with {parallelism} processes")

    # count pre-tokens in parallel
    pre_token_counts: Counter[tuple[bytes]] = Counter()
    with multiprocessing.Pool(parallelism) as pool:
        for result in pool.map(
            functools.partial(_pre_tokenize_from_file_byte_range, file_path, split_special_tokens=split_special_tokens),
            zip(boundaries[:-1], boundaries[1:]),
        ):
            pre_token_counts += result
    return pre_token_counts


def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    for special_token in split_special_tokens:
        assert isinstance(special_token, bytes), "Must represent special token as a bytestring"

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
            found = False
            for special_token in split_special_tokens:
                found_at = mini_chunk.find(special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    found = True
                    break
            if found:
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None):
        self.vocab = vocab
        self.merges = [tuple(m) for m in merges]
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else None
        self._vocab_reverse_lookup: dict[bytes, int] = {v: k for k, v in vocab.items()}

    @classmethod
    def from_file(cls, tokenizer_snapshot: os.PathLike, special_tokens: list[str] | None = None):
        vocab, merges = load_tokenizer(tokenizer_snapshot)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        return list(self.encode_iterable([text]))

    def _encode_special_token(self, token: str) -> int:
        key = token.encode("utf-8", errors="replace")
        return self._vocab_reverse_lookup[key]

    def _encode_regular_text(self, text: str) -> Iterator[int]:
        for m in re.finditer(PRETOKENIZER_PATTERN, text):
            pre_token: list[bytes] = list(x.to_bytes() for x in m.group(0).encode("utf-8"))  # type: ignore

            # apply iterative merges
            for merge_pair in self.merges:
                # cannot merge this pretoken anymore
                if len(pre_token) == 1:
                    break

                updated_pre_token: list[bytes] = []
                prev_token_merged = False
                for pair in itertools.pairwise(pre_token):
                    if not prev_token_merged and pair == merge_pair:
                        updated_pre_token.append(b"".join(merge_pair))
                        prev_token_merged = True
                    else:
                        if prev_token_merged:
                            prev_token_merged = False
                        else:
                            updated_pre_token.append(pair[0])
                if not prev_token_merged:
                    updated_pre_token.append(pair[1])

                pre_token = updated_pre_token

            for elem in pre_token:
                yield self._vocab_reverse_lookup[elem]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        special_token_regex = (
            "|".join([re.escape(s, special_only=True) for s in self.special_tokens]) if self.special_tokens else None
        )
        for data in iterable:
            if special_token_regex:
                last_end = 0
                for m in re.finditer(special_token_regex, data):
                    # Process text before the special token
                    if m.start() > last_end:
                        text_chunk = data[last_end : m.start()]
                        yield from self._encode_regular_text(text_chunk)

                    # Process the special token
                    special_token = m.group()
                    yield self._encode_special_token(special_token)

                    last_end = m.end()

                # Process any remaining text after the last special token
                if last_end < len(data):
                    remaining_text = data[last_end:]
                    yield from self._encode_regular_text(remaining_text)
            else:
                yield from self._encode_regular_text(data)

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[i] for i in ids]).decode("utf-8", errors="replace")
