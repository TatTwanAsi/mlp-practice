import torch
import torch.nn as nn
import torch.optim as optim

def train(model, train_loader, device, epochs = 10, lr = 0.001):
    """
    training function

    Args:
        model(torch.nn.Module):
        train_loader(DataLoader):
        device(string):
        epochs(int):
        lr(float):

    Returns:

    """

    model.to(device)
    model.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)

    for epoch in range(epochs):
        toral_loss = 0.0

        for batch_inputs, batch_labels in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            # forward
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss = loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch[{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')