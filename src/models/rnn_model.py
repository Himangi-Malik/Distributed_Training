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
    model = SimpleRNN()
    return {
        "model": model,
        "criterion": nn.CrossEntropyLoss(),
        "lr": float(config.get("lr", 0.01)),
    }


def train_step(model_obj, config: dict):
    """Execute one training step on synthetic sequence data."""
    model = model_obj["model"]
    criterion = model_obj["criterion"]
    lr = model_obj["lr"]
    
    # Synthetic batch: 8 samples of sequence length 20, vocab size 100
    batch_size = 8
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
    
    # Local SGD update
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param.data -= lr * param.grad
    
    return {
        "rank": config.get("rank", 0),
        "gradients": grad_vector,
        "loss": float(loss),
    }
