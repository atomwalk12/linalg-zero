from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import sympy


class EntropyController:
    """
    Controller for entropy-based mathematical data generation.
    The design follows the implementation at:
    https://github.com/google-deepmind/mathematics_dataset/
    """

    def __init__(self, random_seed: int | None = None):
        """
        Initialize entropy controller.
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def _coprime_density(self, value: int) -> float:
        """Returns asymptotic density of integers coprime to `value`."""
        factors = sympy.factorint(value)
        density = 1.0
        for prime in factors:
            density *= 1 - 1 / prime
        return density

    def generate_integer(
        self, entropy: float, signed: bool = True, min_abs: int = 0, coprime_to: int = 1
    ) -> sympy.Integer:
        """
        Generate random integer with entropy-controlled size. Generates integers
        from a range of size approximately 10^entropy.
        """
        if not isinstance(min_abs, int) or isinstance(min_abs, bool):
            raise TypeError(f"min_abs must be an integer, got {type(min_abs).__name__}")
        coprime_to = abs(coprime_to)
        if min_abs < 0:
            raise ValueError(f"min_abs must be non-negative, got {min_abs}")

        max_ = math.pow(10, entropy)
        max_ += min_abs
        if coprime_to >= 2:
            max_ = max_ / self._coprime_density(coprime_to) + 1

        if signed:
            max_ = math.ceil(max_ / 2)
            range_ = [-max_, max_]
        else:
            max_ = math.ceil(max_)
            range_ = [min_abs, max_]

        while True:
            value = random.randint(*range_)
            if abs(value) >= min_abs and sympy.gcd(value, coprime_to) == 1:
                break

        return sympy.Integer(value)

    def generate_rational(self, entropy: float, signed: bool = True) -> sympy.Rational:
        """Generate a non-integer rational following mathematics_dataset approach."""
        numer_entropy = random.uniform(0, entropy)
        denom_entropy = entropy - numer_entropy
        numer = self.generate_integer(numer_entropy, signed, min_abs=1)
        denom = self.generate_integer(denom_entropy, False, min_abs=2, coprime_to=numer)
        rational = sympy.Rational(numer, denom)
        if isinstance(rational, sympy.Rational):
            return rational
        else:
            raise TypeError(f"This can never happen: {rational}")


def sample_entropy_from_range(entropy_range: tuple[float, float], center_biased_draw: bool = False) -> float:
    """Sample an entropy value from a range with optional center bias.

    When center_biased_draw is True, sample from a symmetric Beta(2,2) and
    scale to [low, high], matching the project's recommended approach.
    """
    low, high = entropy_range
    if not center_biased_draw:
        return random.uniform(low, high)
    if low == high:
        return low
    alpha = 2.0
    x = random.betavariate(alpha, alpha)
    return low + x * (high - low)


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


def uniform_non_negative_integers_with_sum(count: int, sum_: int) -> list[int]:
    """Returns list of size `count` of integers >= 0, summing to `sum_`."""
    positive = uniform_positive_integers_with_sum(count, sum_ + count)
    return [i - 1 for i in positive]


@dataclass(frozen=True)
class SampleArgs:
    """Arguments for sampling mathematical entities with entropy control.

    This class supports sequential composition where components execute in order.
    """

    num_modules: int
    entropy: float

    def peel(self, frac: float = 1.0) -> tuple[float, SampleArgs]:
        """This method provides a fraction of the total entropy budget. It is
        meant to be called iteratively and should be used when we don't know a
        priori the number of required components.
        """
        entropy = frac * self.entropy / self.num_modules
        new_sample_args = SampleArgs(num_modules=self.num_modules, entropy=self.entropy - entropy)
        return entropy, new_sample_args

    def split(self, count: int) -> list[SampleArgs]:
        """
        Splits all available entropy among multiple components using Dirichlet
        distribution.
        """

        # This parameter was modified from the original implementation to ensure
        # all components get meaningful entropy.
        # See: https://github.com/google-deepmind/mathematics_dataset/blob/master/mathematics_dataset/util/composition.py#L90
        num_child_modules = self.num_modules

        # Sample module counts at random - ensure each gets at least some modules
        module_counts = uniform_positive_integers_with_sum(count, num_child_modules)

        # Use Dirichlet distribution for entropy allocation
        entropies = self.entropy * np.random.dirichlet(np.maximum(1e-9, module_counts))

        sample_args = []
        for i, num_modules in enumerate(module_counts):
            child_sample_args = SampleArgs(num_modules=num_modules, entropy=entropies[i])
            sample_args.append(child_sample_args)

        return sample_args
