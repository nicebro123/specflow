"""Sign-invariant encoders for Laplacian eigenvector coordinates."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class SignNet(nn.Module):
    """Encode eigenvector coordinates invariantly under column sign flips."""

    def __init__(
        self,
        n_components: int,
        hidden_dim: int = 32,
        component_dim: int = 4,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if n_components < 1:
            raise ValueError("SignNet requires at least one spectral component")
        self.n_components = n_components
        self.component_dim = component_dim
        self.use_checkpoint = use_checkpoint
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, component_dim),
        )

    @property
    def output_dim(self) -> int:
        return self.n_components * self.component_dim

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, eigenvectors: torch.Tensor) -> torch.Tensor:
        if eigenvectors.shape[-1] != self.n_components:
            raise ValueError(f"expected {self.n_components} spectral components")
        pos = eigenvectors.unsqueeze(-1)
        neg = -pos
        if self.use_checkpoint and self.training and pos.requires_grad:
            encoded = checkpoint(self._encode, pos, use_reentrant=False) + checkpoint(
                self._encode, neg, use_reentrant=False
            )
        else:
            encoded = self.encoder(pos) + self.encoder(neg)
        return encoded.flatten(start_dim=-2)
