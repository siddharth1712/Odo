"""Inference dataset.

Reuses the training ``ChangeLing18KDataset`` (and ``collate_fn``) unchanged, and
adds a ``GenderBatchSampler`` that groups samples by gender so each batch is
single-gender (required by the SMPL estimator, which is gender-conditioned).
"""
import math
import torch
from torch.utils.data import Sampler

from odo.data.dataset import ChangeLing18KDataset, collate_fn  # re-exported for inference scripts


class GenderBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.female_indices = []
        self.male_indices = []
        self.neutral_indices = []

        # Use same logic as the dataset to ensure consistency
        for i, (og_path, ed_path) in enumerate(dataset.final_image_pairs):
            if 'female' in og_path.lower() or 'women' in og_path.lower():
                self.female_indices.append(i)
            elif 'male' in og_path.lower() or 'men' in og_path.lower():
                self.male_indices.append(i)
            else:
                self.neutral_indices.append(i)

        self.batch_size = batch_size

    def __iter__(self):
        for indices in (self.female_indices, self.male_indices, self.neutral_indices):
            for i in range(0, len(indices), self.batch_size):
                yield indices[i:i + self.batch_size]

    def __len__(self):
        female_batches = math.ceil(len(self.female_indices) / self.batch_size) if self.female_indices else 0
        neutral_batches = math.ceil(len(self.neutral_indices) / self.batch_size) if self.neutral_indices else 0
        male_batches = math.ceil(len(self.male_indices) / self.batch_size) if self.male_indices else 0
        return female_batches + neutral_batches + male_batches


if __name__ == "__main__":
    dataset = ChangeLing18KDataset(
        root_dir="/spiral_hdd_2/workspace/siddharth/openpose/FINAL_DATASET_TEST",
        depth_images_path="/spiral_hdd_2/workspace/siddharth/openpose/final_depth_images_test",
        num_samples=None)
    sampler = GenderBatchSampler(dataset, 8)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    print(len(test_dataloader))
