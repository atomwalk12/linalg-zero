import logging

from omegaconf import DictConfig
from torch.utils.data import Dataset
from verl.experimental.dynamic_dataset.dynamicgen_dataset import AbstractDataGenerator

import datasets

logger = logging.getLogger(__name__)


class LinearAlgebraCurriculum(AbstractDataGenerator):
    """
    A noop data gen class that only reappends the first datapoint.
    This class is useful as a placeholder and testing.
    """

    def __init__(self, config: DictConfig = None):
        super().__init__(config)

    def generate(self, dataset: Dataset) -> datasets.Dataset:
        print("MockDataGenerator: No operation performed on the dataset.")
        return dataset.dataframe.select([0])
