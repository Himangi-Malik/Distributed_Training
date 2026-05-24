"""RNN model module for distributed training."""
import torch
import torch.nn as nn


class SimpleRNN(nn.Module):
    """Minimal RNN for benchmark purposes."""
    def __init__(self, input_size=10, hidden_size=32, output_size=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(100, input_size)
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.rnn(x)
        x = h_n.squeeze(0)
        x = self.fc(x)
        return x


def build_model(config: dict):
    """Build and initialize RNN model."""
    # CRITICAL: Set fixed seed on ALL ranks before model creation to ensure
    # identical parameter initialization. This is required for valid distributed SGD.
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    
    model = SimpleRNN()
    lr = float(config.get("lr", 0.01))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    return {
        "model": model,
        "criterion": nn.CrossEntropyLoss(),
        "optimizer": optimizer,
        "lr": lr,
    }


def train_step(model_obj, config: dict):
    """Execute one training step on synthetic sequence data."""
    model = model_obj["model"]
    criterion = model_obj["criterion"]
    lr = model_obj["lr"]
    rank = config.get("rank", 0)
    epoch = config.get("current_epoch", 0)
    step = config.get("step", 0)
    
    # Generate rank-specific synthetic data: different data per rank enables real gradient averaging
    # Seed includes rank and epoch for reproducibility while maintaining data diversity
    data_seed = (
        1000
        + rank * 100000
        + epoch * 1000
        + step
    )
    torch.manual_seed(data_seed)
    
    batch_size = int(config.get("batch_size", 8))
    seq_length = 20
    x = torch.randint(0, 100, (batch_size, seq_length), dtype=torch.long)
    y = torch.randint(0, 5, (batch_size,), dtype=torch.long)
    
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
    
    return {
        "rank": config.get("rank", 0),
        "gradients": grad_vector,
        "loss": float(loss),
    }


def apply_synced_gradients(model_obj, averaged_grad):
    """Apply synchronized averaged gradients back into the model."""
    model = model_obj["model"]
    optimizer = model_obj["optimizer"]
    
    pointer = 0
    for param in model.parameters():
        numel = param.numel()
        param.grad = averaged_grad[pointer:pointer + numel].view_as(param)
        pointer += numel
    
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
