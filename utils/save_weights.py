import os
import torch

def saveWeights(trained_model):
    weights_dir = "./weights"
    os.makedirs(weights_dir, exist_ok=True)

    weights_path = os.path.join(weights_dir, "mlp_weights.pth")
    torch.save(trained_model.state_dict(), weights_path)
    print(f"weights has been saved to: {weights_path}")