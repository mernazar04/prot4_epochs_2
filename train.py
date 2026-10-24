import torch
import torch.nn.functional as F

from torch.utils.data import ( DataLoader, TensorDataset)

from torchvision import (datasets, transforms)

from tqdm import tqdm

from model import make_forward_backward_nets

from bridge import (
    brownian_bridge_sample,
    forward_drift_target,
    backward_drift_target,
    simulate_sde,
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SIGMA = 1.0

# ============================================================
# DATA
# ============================================================

def get_mnist_tensor(n_samples: int) -> torch.Tensor:
    dataset = datasets.MNIST(root="./data", train=True, download=True)
    n_samples = min(n_samples, len(dataset))
    x = dataset.data[:n_samples].float().div(255.0)
    x = (x - 0.5) / 0.5
    return x.unsqueeze(1)

def get_noise_tensor(n_samples: int) -> torch.Tensor:
    return torch.randn(n_samples, 1, 28, 28)


# ============================================================
# INITIAL PAIRS
# ============================================================

def make_independent_pairs(
    x0_pool: torch.Tensor,
    x1_pool: torch.Tensor,
    n: int,
) -> TensorDataset:

    idx0 = torch.randint(0, x0_pool.shape[0], (n,))

    idx1 = torch.randint(0, x1_pool.shape[0], (n,))

    x0 = x0_pool[idx0]
    x1 = x1_pool[idx1]

    return TensorDataset(x0, x1)


# ============================================================
# TRAIN ONE DIRECTION
# ============================================================

def train_direction(
    net: torch.nn.Module,
    pairs: TensorDataset,
    direction: str,
    epochs: int,
    batch_size: int,
    lr: float,
) -> None:

    if direction not in (
        "forward",
        "backward",
    ):
        raise ValueError(
            "direction должен быть "
            "'forward' или 'backward'"
        )

    loader = DataLoader(
        pairs,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=lr,
    )

    net.train()

    for epoch in range(epochs):

        running_loss = 0.0

        pbar = tqdm(
            loader,
            desc=(
                f"[{direction}] "
                f"epoch {epoch + 1}/{epochs}"
            ),
        )

        for x0, x1 in pbar:

            x0 = x0.to(DEVICE)
            x1 = x1.to(DEVICE)

            # t ~ Uniform(0, 1)
            t = torch.empty( x0.shape[0], device=DEVICE, ).uniform_( 1e-3, 1.0 - 1e-3)

            x_t = brownian_bridge_sample( x0, x1, t, sigma=SIGMA)

            if direction == "forward":
                target = forward_drift_target(x1, x_t, t)

            else:
                target = backward_drift_target(x0, x_t, t)

            
            #=======================================================================
            # prediction = net(x_t, t)
            # loss = F.mse_loss(prediction, target)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            prediction = net(x_t, t)
            w = (1 - t).view(-1, 1, 1, 1) if direction == "forward" else t.view(-1, 1, 1, 1)
            loss = F.mse_loss(prediction * w, target * w)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            #=======================================================================
            
            running_loss += loss.item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        average_loss = ( running_loss / len(loader))

        #============================================================
        # print(
        #     f"[{direction}] "
        #     f"epoch {epoch + 1}: "
        #     f"avg_loss={average_loss:.6f}"
        # )
        print(
            f"[{direction}] "
            f"epoch {epoch + 1}: "
            f"avg_loss={average_loss:.6f}"
        )

        torch.save(net.state_dict(), f"dsb_{direction}.pt")
        #===============================================================


# ============================================================
# REGENERATE PAIRS
# ============================================================

@torch.no_grad()
def regenerate_pairs(
    net: torch.nn.Module,
    x_start_pool: torch.Tensor,
    direction: str,
    n: int,
    num_steps: int,
    batch_size: int = 256,
) -> TensorDataset:

    net.eval()

    idx = torch.randint(0, x_start_pool.shape[0], (n,))

    x_start = x_start_pool[idx]

    ends = []

    for i in range( 0, n, batch_size ):

        chunk = x_start[
            i:i + batch_size
        ].to(DEVICE)

        x_end = simulate_sde(
            net,
            chunk,
            direction=direction,
            num_steps=num_steps,
            sigma=SIGMA,
        )

        ends.append(
            x_end.cpu()
        )

    x_end = torch.cat(ends, dim=0)

    if direction == "forward":
        # x0 -> x1_hat
        return TensorDataset(x_start, x_end)

    else:
        # x0_hat -> x1
        return TensorDataset(x_end, x_start)


# ============================================================
# IPF
# ============================================================

def ipf_outer_loop(
    n_ipf_iters: int = 1,
    n_pairs: int = 10000,
    inner_epochs: int = 30,
    batch_size: int = 128,
    lr: float = 2e-4,
    sde_steps: int = 50,
    ckpt_prefix: str = "dsb",
) -> None:

    print(f"Устройство: {DEVICE}")

    p0_pool = get_noise_tensor(60000)
    p1_pool = get_mnist_tensor(60000)

    print(f"p0 = Gaussian: {p0_pool.shape}")

    print(f"p1 = MNIST:    {p1_pool.shape}")

    # --------------------------------------------------------
    # Создаём две сети
    # --------------------------------------------------------

    forward_net, backward_net = (make_forward_backward_nets())

    forward_net.to(DEVICE)
    backward_net.to(DEVICE)

    # --------------------------------------------------------
    # Начальное coupling:
    #
    #       p0(x0) p1(x1)
    # --------------------------------------------------------

    pairs = make_independent_pairs(p0_pool, p1_pool, n_pairs)

    # ========================================================
    # IPF iterations
    # ========================================================

    for k in range(n_ipf_iters):

        print(
            f"\n========== "
            f"IPF {k + 1}/{n_ipf_iters} "
            f"=========="
        )

        # ----------------------------------------------------
        # 1. Forward
        #
        #       p0 -> p1
        # ----------------------------------------------------

        #===================================================================
        # train_direction(
        #     forward_net,
        #     pairs,
        #     direction="forward",
        #     epochs=inner_epochs,
        #     batch_size=batch_size,
        #     lr=lr,
        # )
        train_direction(
            forward_net,
            pairs,
            direction="forward",
            epochs=inner_epochs,
            batch_size=batch_size,
            lr=lr,
        )

        torch.save(
            forward_net.state_dict(),
            f"{ckpt_prefix}_forward.pt",
        )

        if n_ipf_iters == 1:
            break
        #================================================================

        # Пропускаем Gaussian noise через
        # обученную forward SDE.
        pairs = regenerate_pairs(
            forward_net,
            p0_pool,
            direction="forward",
            n=n_pairs,
            num_steps=sde_steps,
        )

        print(f"forward pairs: {len(pairs)}")

        # ----------------------------------------------------
        # 2. Backward
        #
        #       p1 -> p0
        # ----------------------------------------------------

        train_direction(
            backward_net,
            pairs,
            direction="backward",
            epochs=inner_epochs,
            batch_size=batch_size,
            lr=lr,
        )

        pairs = regenerate_pairs(
            backward_net,
            p1_pool,
            direction="backward",
            n=n_pairs,
            num_steps=sde_steps,
        )

        print(f"backward pairs: {len(pairs)}")

        # ----------------------------------------------------
        # Checkpoints
        # ----------------------------------------------------

        torch.save(
            forward_net.state_dict(),
            f"{ckpt_prefix}_forward.pt",
        )

        torch.save(
            backward_net.state_dict(),
            f"{ckpt_prefix}_backward.pt",
        )

    print("\nОбучение завершено.")

    print(
        f"Forward: "
        f"{ckpt_prefix}_forward.pt"
    )

    print(
        f"Backward: "
        f"{ckpt_prefix}_backward.pt"
    )

if __name__ == "__main__":
    ipf_outer_loop()