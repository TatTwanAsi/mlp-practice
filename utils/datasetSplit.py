import torch
from torch.utils.data import random_split

def datasetSplit(dataset, ratio = 0.8):
    """
    split the dataset into training and test sets by ratio.

    Args:
        dataset(torch.utils.data.TensorDataset): the dataset.
        ratio(float): the training set proportion of the total dataset.
    
    Returns:

    
    """
    total_size = len(dataset)
    train_size = int(ratio * total_size)
    test_size = total_size - train_size

    return random_split(dataset, [train_size, test_size])