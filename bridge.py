import torch


def brownian_bridge_sample(
    x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, sigma: float = 1.0
) -> torch.Tensor:

    t_ = t.view(-1, 1, 1, 1)
    mean = (1 - t_) * x0 + t_ * x1
    var = (t_ * (1 - t_)) * (sigma**2)
    std = torch.sqrt(var.clamp_min(1e-8))
    eps = torch.randn_like(x0)
    return mean + std * eps


def forward_drift_target(x1: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    t_ = t.view(-1, 1, 1, 1)
    return (x1 - x_t) / (1 - t_).clamp_min(1e-4)


def backward_drift_target(x0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    t_ = t.view(-1, 1, 1, 1)
    return (x0 - x_t) / t_.clamp_min(1e-4)


@torch.no_grad()
def simulate_sde(net, x_start, direction, num_steps=50, sigma=1.0):
    assert direction in ("forward", "backward")
    x = x_start.clone()
    B = x.shape[0]
    dt = 1.0 / num_steps
    ks = torch.arange(num_steps, device=x.device)
    t_values = ks * dt if direction == "forward" else 1.0 - ks * dt

    for k, t_scalar in enumerate(t_values):
        t = torch.full((B,), float(t_scalar), device=x.device).clamp(1e-3, 1 - 1e-3)
        x = x + net(x, t) * dt                  # знак всегда +
        if k < num_steps - 1:                   # на последнем шаге без шума
            x = x + sigma * dt**0.5 * torch.randn_like(x)
    return x