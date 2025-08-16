import logging

from torch.utils.data import Dataset
from verl.experimental.dynamic_dataset.dynamicgen_dataset import AbstractDataGenerator

from datasets import Dataset as HFDataset

logger = logging.getLogger(__name__)


class LinearAlgebraCurriculum(AbstractDataGenerator):
    """
    A noop data gen class that only reappends the first datapoint.
    This class is useful as a placeholder and testing.
    """

    def __init__(self, dataset: HFDataset = None):
        self.dataset = dataset

    def generate(self, dataset: Dataset) -> HFDataset:
        print("MockDataGenerator: No operation performed on the dataset.")
        return self.dataset
