"""CNN model module for distributed training."""
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Minimal CNN for benchmark purposes."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def build_model(config: dict):
    """Build and initialize CNN model."""
    model = SimpleCNN()
    return {
        "model": model,
        "criterion": nn.CrossEntropyLoss(),
        "lr": float(config.get("lr", 0.01)),
    }


def train_step(model_obj, config: dict):
    """Execute one training step on synthetic data."""
    model = model_obj["model"]
    criterion = model_obj["criterion"]
    lr = model_obj["lr"]
    
    # Synthetic batch: 4 samples of 28x28 single-channel images
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28, dtype=torch.float32)
    y = torch.randint(0, 10, (batch_size,), dtype=torch.long)
    
    # Forward pass
    output = model(x)
    loss = criterion(output, y)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Collect gradients into a single vector
    grad_list = []
    for param in model.parameters():
        if param.grad is not None:
            grad_list.append(param.grad.detach().clone().flatten())
    
    grad_vector = torch.cat(grad_list) if grad_list else torch.tensor([0.0], dtype=torch.float32)
    
    # Local SGD update
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param.data -= lr * param.grad
    
    return {
        "rank": config.get("rank", 0),
        "gradients": grad_vector,
        "loss": float(loss.detach()),
    }
