import itertools
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import data.bucketing as bucketing


class DummyDataset:
    column_names = ["target_resolution"]

    def __init__(self, resolutions):
        self._resolutions = [tuple(map(int, res)) for res in resolutions]

    def __len__(self):
        return len(self._resolutions)

    def __getitem__(self, item):
        if isinstance(item, str):
            if item == "target_resolution":
                return [list(res) for res in self._resolutions]
            raise KeyError(item)
        res = self._resolutions[item]
        return {"target_resolution": res}


@pytest.mark.parametrize("drop_last", [True, False])
def test_bucket_sampler_partitions_batches_across_ranks(monkeypatch, drop_last):
    rng = random.Random(42)
    bucket_pool = bucketing.aspect_ratio_buckets_256
    resolutions = [rng.choice(bucket_pool) for _ in range(100)]
    dataset = DummyDataset(resolutions)
    aspect_buckets = bucketing.aspect_ratio_buckets_256
    batch_size = 2
    world_size = 2

    samplers = []
    counts = []
    for rank in range(world_size):
        monkeypatch.setattr(bucketing, "_world_info", lambda r=rank: (world_size, r))
        sampler = bucketing.BucketBatchSampler(
            dataset,
            aspect_ratio_buckets=aspect_buckets,
            batch_size=batch_size,
            drop_last=drop_last,
            base_seed=123,
        )
        sampler.set_epoch(0)
        samplers.append(sampler)

    all_batches = []
    for sampler in samplers:
        batches = list(sampler)
        assert len(batches) == len(sampler)
        all_batches.append(batches)
        counts.append(len(batches))

    for rank, batches in enumerate(all_batches):
        for step, batch in enumerate(batches):
            print(f"drop_last={drop_last} rank={rank} step={step} batch={batch}")

    flat_rank_batches = [list(itertools.chain.from_iterable(batches)) for batches in all_batches]

    if drop_last:
        assert set(flat_rank_batches[0]).isdisjoint(flat_rank_batches[1])

    # union of batches across ranks should cover the expected number of samples
    combined = flat_rank_batches[0] + flat_rank_batches[1]
    print(f"drop_last={drop_last} combined_indices_sorted={sorted(combined)}")
    total_items = len(resolutions)
    expected_total = sum(count * batch_size for count in counts)
    assert len(combined) == expected_total
    assert set(combined).issubset(set(range(total_items)))
    if not drop_last:
        # every sample should appear at least once
        assert set(range(total_items)).issubset(set(combined))


def test_bucket_sampler_shuffles_each_epoch(monkeypatch):
    rng = random.Random(7)
    bucket_pool = bucketing.aspect_ratio_buckets_512
    dataset = DummyDataset([rng.choice(bucket_pool) for _ in range(200)])
    monkeypatch.setattr(bucketing, "_world_info", lambda: (1, 0))

    sampler = bucketing.BucketBatchSampler(
        dataset,
        aspect_ratio_buckets=bucket_pool,
        batch_size=4,
        drop_last=True,
        base_seed=314,
    )

    def flatten_epoch(cur_sampler: bucketing.BucketBatchSampler) -> list[int]:
        batches = list(cur_sampler)
        assert len(batches) == len(cur_sampler)
        return list(itertools.chain.from_iterable(batches))

    sampler.set_epoch(0)
    epoch0_order = flatten_epoch(sampler)
    sampler.set_epoch(1)
    epoch1_order = flatten_epoch(sampler)

    # Sanity-check deterministic repeats for the same epoch
    sampler_repeat = bucketing.BucketBatchSampler(
        dataset,
        aspect_ratio_buckets=bucket_pool,
        batch_size=4,
        drop_last=True,
        base_seed=314,
    )
    sampler_repeat.set_epoch(0)
    repeat_order = flatten_epoch(sampler_repeat)
    print("E0",epoch0_order)
    print("E1",epoch1_order)
    assert repeat_order == epoch0_order
    assert epoch1_order != epoch0_order
