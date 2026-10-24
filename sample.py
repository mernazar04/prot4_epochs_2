import torch
from torchvision.utils import save_image

from model import DriftNet
from bridge import simulate_sde


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CHECKPOINT = "dsb_forward.pt"


@torch.no_grad()
def sample_from_noise(
    x: torch.Tensor,
    checkpoint: str = CHECKPOINT,
    num_steps: int = 100,
    sigma: float = 1.0,
):
    
    if x.ndim != 4:
        raise ValueError(
            "x должен иметь форму "
            "[B, 1, 28, 28]"
        )

    if x.shape[1:] != (1, 28, 28):
        raise ValueError(
            "Ожидается форма "
            "[B, 1, 28, 28]"
        )

    net = DriftNet(image_size=28, base_channels=32).to(DEVICE)

    state_dict = torch.load(checkpoint, map_location=DEVICE)

    net.load_state_dict(state_dict)

    net.eval()

    x_initial = x.to(DEVICE).clone()

    x_final = simulate_sde(
        net,
        x_initial,
        direction="forward",
        num_steps=num_steps,
        sigma=sigma,
    )

    return (x_initial, x_final)


def main():

    x = torch.randn(1, 1, 28, 28, device=DEVICE)

    x_initial, x_final = (
        sample_from_noise(
            x,
            checkpoint="dsb_forward.pt",
            num_steps=100,
            sigma=1.0,
        )
    )

    initial_image = (x_initial.clamp(-1, 1) + 1) / 2

    final_image = (x_final.clamp(-1, 1) + 1) / 2

    save_image(initial_image, "initial_noise.png")

    save_image(final_image, "generated.png")

    print("Initial:", tuple(x_initial.shape))

    print("Final:", tuple(x_final.shape))

    print("Saved: initial_noise.png")

    print("Saved: generated.png")


if __name__ == "__main__":
    main()


# ================================================================================================================================================================
# ================================================================================================================================================================
# ================================================================================================================================================================
# ================================================================================================================================================================



# import argparse
# import torch
# import torchvision.utils as vutils

# from model import DriftNet
# from bridge import simulate_sde
# from train import get_class_tensor, get_noise_pool, DEVICE, SIGMA


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--p1_type", choices=["digit", "noise"], default="digit")
#     parser.add_argument("--direction", choices=["forward", "backward"], default="forward")
#     parser.add_argument("--n", type=int, default=16)
#     parser.add_argument("--sde_steps", type=int, default=50)
#     parser.add_argument("--ckpt_prefix", type=str, default="dsb")
#     parser.add_argument("--out", type=str, default="samples.png")
#     args = parser.parse_args()

#     net = DriftNet().to(DEVICE)
#     ckpt = f"{args.ckpt_prefix}_{args.direction}.pt"
#     net.load_state_dict(torch.load(ckpt, map_location=DEVICE))
#     net.eval()
#     print(f"Загружены веса: {ckpt}")

#     # старт: реальные сэмплы с ТОГО конца, откуда идём
#     if args.p1_type == "digit":
#         start_digit = 0 if args.direction == "forward" else 1
#         x_start = get_class_tensor(digit=start_digit, n_samples=args.n).to(DEVICE)
#     else:
#         if args.direction == "forward":
#             x_start = get_class_tensor(digit=None, n_samples=args.n).to(DEVICE)
#         else:
#             # direction="backward": старт из ЧИСТОГО ШУМА - это и есть генерация
#             x_start = get_noise_pool(n_samples=args.n).to(DEVICE)

#     x_end = simulate_sde(
#         net, x_start, direction=args.direction, num_steps=args.sde_steps, sigma=SIGMA
#     )

#     grid_in = vutils.make_grid((x_start.cpu() + 1) / 2, nrow=4)
#     grid_out = vutils.make_grid((x_end.cpu() + 1) / 2, nrow=4)
#     vutils.save_image(grid_in, f"input_{args.out}")
#     vutils.save_image(grid_out, args.out)
#     print(f"Сохранено: input_{args.out} (вход) и {args.out} (результат моста)")


# if __name__ == "__main__":
#     main()
