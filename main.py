from utils.train import train
from utils.dataGenerator import dataGenerator
from utils.datasetSplit import datasetSplit
from torch.utils.data import DataLoader
from model.mlp import MLP
import torch
import torch.nn as nn

def main():

    # use CUDA if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    # prepare data
    dataset = dataGenerator()
    train_set, test_set = datasetSplit(dataset)
    train_loader = DataLoader(train_set, batch_size = 128, shuffle = True)
    test_loader = DataLoader(test_set, batch_size = 128, shuffle = True)

    # model
    model = MLP()

    train(model, train_loader, device)


if __name__ == "__main__":
    main()