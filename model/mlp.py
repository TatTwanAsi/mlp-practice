import torch

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.FC = torch.nn.Linear(2, 1)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.FC(x)
        x = self.relu(x)
        return x
    