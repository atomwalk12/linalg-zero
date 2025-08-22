import math
import random

import numpy as np
import sympy

from linalg_zero.shared.utils import get_logger

logger = get_logger(__name__)


class EntropyController:
    """
    Controller for entropy-based mathematical data generation.
    The design follows the implementation at:
    https://github.com/google-deepmind/mathematics_dataset/
    """

    def __init__(self, total_entropy: float = 3.0, random_seed: int | None = None):
        """
        Initialize entropy controller with total entropy budget.
        """
        self.total_entropy = total_entropy
        self.remaining_entropy = total_entropy
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

    def generate_rational(
        self, entropy: float, signed: bool = True, force_non_integer: bool = False
    ) -> sympy.Rational:
        """Generate random rational number with entropy-controlled complexity."""
        if force_non_integer:
            return self._generate_non_integer_rational(entropy, signed)
        else:
            # 50% probability of returning integer vs non-integer rational
            if random.choice([False, True]):
                return sympy.Rational(self.generate_integer(entropy, signed))  # type: ignore[return-value]
            else:
                return self._generate_non_integer_rational(entropy, signed)

    def _generate_non_integer_rational(self, entropy: float, signed: bool) -> sympy.Rational:
        """Generate a non-integer rational following mathematics_dataset approach."""
        numer_entropy = random.uniform(0, entropy)
        denom_entropy = entropy - numer_entropy
        numer = self.generate_integer(numer_entropy, signed, min_abs=1)
        denom = self.generate_integer(denom_entropy, False, min_abs=2, coprime_to=numer)
        return sympy.Rational(numer, denom)  # type: ignore[return-value]
