"""ANN model module for distributed training benchmarks."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class SimpleANN(nn.Module):
    """Small fully connected network for Fashion-MNIST benchmark runs."""

    def __init__(self, input_dim: int = 28 * 28, hidden_dim: int = 128, output_dim: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def _load_next_batch(state):
    try:
        return next(state["iterator"])
    except StopIteration:
        state["iterator"] = iter(state["loader"])
        return next(state["iterator"])


def _flatten_gradients(model):
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.detach().clone().flatten())
    return torch.cat(grads) if grads else torch.tensor([0.0], dtype=torch.float32)


def build_model(config: dict):
    """Build and initialize ANN model state."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[2]
    data_root = Path(config.get("data_root", project_root / "data"))

    model = SimpleANN(
        input_dim=int(config.get("ann_input_dim", 28 * 28)),
        hidden_dim=int(config.get("ann_hidden_dim", 128)),
        output_dim=int(config.get("ann_output_dim", 10)),
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=float(config.get("lr", 0.01)))
    criterion = nn.CrossEntropyLoss()
    transform = transforms.ToTensor()
    dataset = datasets.FashionMNIST(
        root=str(data_root),
        train=True,
        download=False,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", config.get("ann_batch_size", 64))),
        shuffle=True,
    )

    return {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "loader": loader,
        "device": device,
        "iterator": iter(loader),
        "lr": float(config.get("lr", 0.01)),
    }


def train_step(state, config: dict):
    """Execute one Fashion-MNIST training step."""
    model = state["model"]
    optimizer = state["optimizer"]
    criterion = state["criterion"]
    device = state["device"]

    x, y = _load_next_batch(state)
    x = x.view(x.size(0), -1).to(device)
    y = y.to(device)

    model.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    grad_vector = _flatten_gradients(model)

    return {
        "rank": config.get("rank", 0),
        "gradients": grad_vector,
        "loss": float(loss.detach()),
    }


def apply_synced_gradients(state, averaged_grad):
    """Write averaged gradients back into the model and step the optimizer."""
    model = state["model"]
    optimizer = state["optimizer"]

    pointer = 0
    for param in model.parameters():
        numel = param.numel()
        param.grad = averaged_grad[pointer:pointer + numel].view_as(param)
        pointer += numel

    optimizer.step()
