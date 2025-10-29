# from utils.train import train
from utils.dataGenerator import dataGenerator
from utils.datasetSplit import datasetSplit
import torch

def main():

    # use CUDA if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    # prepare data
    dataset = dataGenerator()
    train_set, test_set = datasetSplit(dataset)
    for data in train_set:
        print(data)

    print(dataset)
    # print(test_set)


if __name__ == "__main__":
    main()