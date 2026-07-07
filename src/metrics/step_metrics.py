from dataclasses import asdict, dataclass


@dataclass(slots=True)
class StepMetrics:
    """
    Metrics collected for one training iteration on a single rank.
    """

    # Step Information
    epoch: int
    step: int
    is_warmup: bool

    # Timing Metrics (seconds)
    compute_time: float
    sync_time: float
    optim_time: float
    iteration_time: float

    # Model Metrics
    loss: float
    grad_norm: float

    # Communication Metrics (bytes)
    bytes_sent: int
    bytes_received: int

    def to_dict(self) -> dict:
        return asdict(self)
    

    @classmethod
    def from_dict(cls, data: dict) -> "StepMetrics":
        if "iteration_time" not in data:
            data = dict(data)
            data["iteration_time"] = (
                data["compute_time"]
                + data["sync_time"]
                + data["optim_time"]
            )

        return cls(**data)  