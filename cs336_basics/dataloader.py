import hashlib
import grain.python as grain
import glob

import numpy as np


class ToTrainingExamples(grain.experimental.FlatMapTransform):
    max_fan_out = 10_000

    def __init__(self, context_length: int, samples_per_document: int | None):
        self.context_length = context_length
        self.samples_per_document = samples_per_document

    def flat_map(self, elem):
        tokens = np.frombuffer(elem, dtype=np.uint16)

        # valid_ranges
        valid_ranges = []
        for i in range(0, len(tokens) - self.context_length):
            first_input = i
            last_input = i + self.context_length
            first_target = i + 1
            last_target = i + self.context_length + 1
            valid_ranges.append((first_input, last_input, first_target, last_target))

        # Example: randomly shuffle or sample
        indices = np.arange(len(valid_ranges))
        if self.samples_per_document is not None:
            hash_obj = hashlib.sha256(elem)
            seed = int.from_bytes(hash_obj.digest()[:4], "little")
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
            indices = indices[: self.samples_per_document]

        for idx in indices:
            a, b, c, d = valid_ranges[idx]
            yield (tokens[a:b], tokens[c:d])


def get_data_loader_lazy(
    dataset_path,
    sequence_length: int = 1024,
    samples_per_file: int = 1000,
    batch_size: int = 1,
    num_epochs: int = 1,
    seed: int = 43,
):
    source = grain.ArrayRecordDataSource(glob.glob(dataset_path))

    index_sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=num_epochs,
        shuffle=True,
        seed=seed,
    )

    def _get_data_loader():
        return grain.DataLoader(
            data_source=source,
            operations=[
                ToTrainingExamples(sequence_length, samples_per_file),
                grain.Batch(batch_size=batch_size, drop_remainder=True),
            ],
            sampler=index_sampler,
            worker_count=1,
        )

    return _get_data_loader
