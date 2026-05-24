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
    # CRITICAL: Set fixed seed on ALL ranks before model creation to ensure
    # identical parameter initialization. This is required for valid distributed SGD.
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    
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
    rank = config.get("rank", 0)
    epoch = config.get("current_epoch", 0)
    
    # Generate rank-specific synthetic data: different data per rank enables real gradient averaging
    # Seed includes rank and epoch for reproducibility while maintaining data diversity
    data_seed = 1000 + rank * 100 + epoch
    torch.manual_seed(data_seed)
    
    batch_size = int(config.get("batch_size", 4))
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


def apply_synced_gradients(model_obj, averaged_grad):
    """Apply synchronized averaged gradients back into the model."""
    model = model_obj["model"]
    
    pointer = 0
    for param in model.parameters():
        numel = param.numel()
        param.grad = averaged_grad[pointer:pointer + numel].view_as(param)
        pointer += numel
