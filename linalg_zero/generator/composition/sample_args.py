from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


def uniform_positive_integers_with_sum(count: int, sum_: int) -> list[int]:
    """Returns list of size `count` of integers >= 1, summing to `sum_`."""
    if sum_ < 0:
        raise ValueError(f"Sum must be non-negative, got {sum_}")
    if count > sum_:
        raise ValueError(f"Cannot find {count} numbers >= 1 with sum {sum_}")
    if count == 0:
        return []
    # Select `count - 1` numbers from {1, ..., sum_ - 1}
    separators = random.sample(list(range(1, sum_)), count - 1)
    separators = sorted(separators)
    return [right - left for left, right in zip([0, *separators], [*separators, sum_], strict=False)]


def uniform_non_negative_integers_with_sum(count, sum_):
    """Returns list of size `count` of integers >= 0, summing to `sum_`."""
    positive = uniform_positive_integers_with_sum(count, sum_ + count)
    return [i - 1 for i in positive]


@dataclass(frozen=True)
class SampleArgs:
    """Arguments for sampling mathematical entities with entropy control.

    This class supports sequential composition where components execute in order.
    The split() method distributes entropy among all components simultaneously
    using Dirichlet distribution. The design follows the implementation at:
    https://github.com/google-deepmind/mathematics_dataset/blob/master/mathematics_dataset/util/composition.py#L90
    """

    num_modules: int
    entropy: float

    def split(self, count: int) -> list[SampleArgs]:
        """Splits entropy among multiple components using Dirichlet distribution.

        Args:
            count: Number of components to split entropy among
            weights: Optional weights for components (defaults to uniform)
            min_entropy_per_component: Minimum entropy each component should receive

        Returns:
            List of SampleArgs for each component

        Raises:
            ValueError: If entropy cannot be fully allocated
        """
        num_child_modules = self.num_modules - 1

        # Sample module counts at random
        module_counts = uniform_non_negative_integers_with_sum(count, num_child_modules)

        if num_child_modules == 0:
            if self.entropy > 0:
                raise ValueError("Unused entropy")
            entropies = np.zeros(count)
        else:
            entropies = self.entropy * np.random.dirichlet(np.maximum(1e-9, module_counts))

        sample_args = []
        for i, num_modules in enumerate(module_counts):
            child_sample_args = SampleArgs(num_modules=num_modules, entropy=entropies[i])
            sample_args.append(child_sample_args)

        return sample_args


# NOTE[Future]: PreSampleArgs could be added here for randomized entropy ranges
# if needed for curriculum learning or problem variety. For now, we use
# concrete SampleArgs values for simplicity and predictability.
