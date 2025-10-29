import torch

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.FC1 = torch.nn.Linear(2, 64)
        self.FC2 = torch.nn.Linear(64, 32)
        self.FC3 = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.FC1(x))
        x = self.relu(self.FC2(x))
        x = self.FC3(x)
        return x
    