import torch
from torch.utils.data import TensorDataset
import random

def dataGenerator(sample_num = 10000, min_limit = -100, max_limit = 100):

    """
        generate dataset for training

        Args:
            sample_num(int): number of data samples to generate for the dataset. Default is 10,000.
            min_limit(int): minimum possible value for a or b. Default is -100.
            max_limit(int): maximum possible value for a or b. Default is 100.

        Returns:
            inputs_tensor: a tensor of shape(sample_num, 2), where each line is (a, b).
            labels_tensor: a tensor of shape(sample_num, 1), where each line is the corresponding a+b.
    """

    inputs_tensor = torch.rand(sample_num, 2, dtype = torch.float32) * (max_limit - min_limit) + min_limit
    labels_tensor = inputs_tensor.sum(dim=1).unsqueeze(1)

    return TensorDataset(inputs_tensor, labels_tensor)
