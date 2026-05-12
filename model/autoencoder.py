import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    Bottleneck autoencoder for anomaly detection on 200-dim HVAC sensor windows.
    Trained only on normal data; faults produce high reconstruction error.

    Architecture: 200 -> 128 -> 64 -> 8 -> 64 -> 128 -> 200
    """

    def __init__(self, input_dim: int = 200, bottleneck: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_errors(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE between input and reconstruction (no grad)."""
        self.eval()
        with torch.no_grad():
            recon = self(x)
            return torch.mean((x - recon) ** 2, dim=1)


def load_autoencoder(checkpoint_path: str) -> "Autoencoder":
    """Load a saved autoencoder checkpoint and return the model in eval mode."""
    ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = Autoencoder(
        input_dim  = ckpt.get("input_dim", 200),
        bottleneck = ckpt.get("bottleneck", 8),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model
