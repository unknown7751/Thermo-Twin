import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    Bottleneck autoencoder for anomaly detection on 200-dim HVAC sensor windows.
    Trained only on normal data; faults produce high reconstruction error.

    Architecture: 200 -> 128 -> 64 -> 8 -> 64 -> 128 -> 200
    Dropout(p) after each hidden ReLU enables MC-Dropout uncertainty estimation.
    """

    def __init__(self, input_dim: int = 200, bottleneck: int = 8, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_errors(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE between input and reconstruction (no grad, eval mode)."""
        self.eval()
        with torch.no_grad():
            recon = self(x)
            return torch.mean((x - recon) ** 2, dim=1)

    def mc_reconstruction_errors(self, x: torch.Tensor, n_passes: int = 10) -> torch.Tensor:
        """
        N stochastic forward passes with dropout active.
        Returns tensor of shape (n_passes, N) — one row of per-sample errors per pass.
        """
        self.train()
        with torch.no_grad():
            passes = torch.stack([
                torch.mean((x - self(x)) ** 2, dim=1)
                for _ in range(n_passes)
            ])
        self.eval()
        return passes


def load_autoencoder(checkpoint_path: str) -> "Autoencoder":
    """Load a saved autoencoder checkpoint and return the model in eval mode."""
    ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = Autoencoder(
        input_dim  = ckpt.get("input_dim", 200),
        bottleneck = ckpt.get("bottleneck", 8),
        dropout    = ckpt.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model