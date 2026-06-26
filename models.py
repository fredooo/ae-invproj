import os
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn

from config import Config, LayerConfig


def build_sequential(
    layers: List[LayerConfig], input_shape: int | Tuple[int, int, int]
) -> Tuple[nn.Sequential, int | Tuple[int, int, int]]:
    seq_layers = []
    current_shape: int | Tuple[int, int, int] = input_shape

    for layer_cfg in layers:
        layer_module, out_shape = layer_cfg.to_module(current_shape)
        seq_layers.append(layer_module)
        if out_shape is not None:
            current_shape = out_shape  # update for next layer

    return nn.Sequential(*seq_layers), current_shape


class Encoder(ABC):
    @abstractmethod
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input tensor into a lower-dimensional representation."""
        pass


class Decoder(ABC):
    @abstractmethod
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode input tensor into a higher-dimensional representation."""
        pass


class EncoderBase(nn.Module):
    def __init__(self, io_dim: int, layers: List[LayerConfig]) -> None:
        super().__init__()
        self.io_dim = io_dim
        layers_seq, out_shape = build_sequential(layers, io_dim)
        self.net = layers_seq
        if isinstance(out_shape, int):
            self.last_hidden_dim = out_shape
        else:
            raise ValueError(f"Invalid shape {out_shape}: expected flattned input (int) as output of the sequential")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.io_dim)
        x = self.net(x)
        return x


class HeadSimple(nn.Module):
    def __init__(self, last_hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(last_hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        return self.fc(x), None, None


class HeadGaussianFull(nn.Module):
    def __init__(self, last_hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.fc_mu = nn.Linear(last_hidden_dim, latent_dim)
        self.fc_L_param = nn.Linear(last_hidden_dim, latent_dim * (latent_dim + 1) // 2)

    @staticmethod
    def construct_L(L_params, latent_dim):
        batch_size = L_params.size(0)
        L = torch.zeros(batch_size, latent_dim, latent_dim, device=L_params.device)
        tril_idx = torch.tril_indices(latent_dim, latent_dim)
        L[:, tril_idx[0], tril_idx[1]] = L_params
        diag_indices = torch.arange(latent_dim)
        L[:, diag_indices, diag_indices] = torch.exp(L[:, diag_indices, diag_indices])
        return L

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.fc_mu(x)
        L_params = self.fc_L_param(x)
        L = self.construct_L(L_params, self.latent_dim)

        if self.training:
            eps = torch.randn(mu.size(0), self.latent_dim, 1, device=x.device)
            z = mu.unsqueeze(2) + torch.bmm(L, eps)
            z = z.squeeze(2)
        else:
            z = mu

        return z, mu, L


class HeadGaussianDiagonal(nn.Module):
    def __init__(self, last_hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.fc_mu = nn.Linear(last_hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(last_hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        return z, mu, logvar


class HeadGaussianIsotropic(nn.Module):
    def __init__(self, last_hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.fc_mu = nn.Linear(last_hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(last_hidden_dim, 1)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.fc_mu(x)
        logvar_scalar = self.fc_logvar(x)
        logvar = logvar_scalar.expand(-1, self.latent_dim)

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        return z, mu, logvar


class DecoderNetwork(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        layers: List[LayerConfig],
    ) -> None:
        super().__init__()

        self.net, _ = build_sequential(layers, latent_dim)

    def forward(self, z) -> torch.Tensor:
        return self.net(z)


class EncoderDecoder(nn.Module, Encoder, Decoder):
    encoder_head: nn.Module

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        self.encoder_base = EncoderBase(config.model.io_dim, config.model.encoder_layers)

        self.decoder = DecoderNetwork(config.model.latent_dim, config.model.decoder_layers)

    @abstractmethod
    def encode(self, x) -> torch.Tensor:
        pass

    def decode(self, z):
        return self.decoder(z)

    @abstractmethod
    def forward(self, x):
        pass


class ProjectorReconstructor(EncoderDecoder):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.encoder_head = HeadSimple(self.encoder_base.last_hidden_dim, config.model.latent_dim)

    def encode(self, x):
        x = self.encoder_base(x)
        z, _, _ = self.encoder_head(x)
        return z, None, None

    def forward(self, original: torch.Tensor, proj_target: torch.Tensor, mode: str = "full"):
        if mode is None or mode == "full":
            z, _, _ = self.encode(original)
            recon = self.decoder(z)
            return recon, z, None, None
        elif mode == "encode":
            z, _, _ = self.encode(original)
            return None, z, None, None
        elif mode == "decode":
            return self.decode(proj_target), None, None, None
        else:
            raise ValueError(f"Unknown mode: {mode}")


class AE(EncoderDecoder):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.encoder_head = HeadSimple(self.encoder_base.last_hidden_dim, config.model.latent_dim)

    def encode(self, x):
        x = self.encoder_base(x)
        z, _, _ = self.encoder_head(x)
        return z, None, None

    def forward(self, original: torch.Tensor, proj_target: torch.Tensor, mode: str = "full"):
        x = self.encoder_base(original)
        z, _, _ = self.encoder_head(x)
        recon = self.decoder(z)
        return recon, z, None, None


class VAE(EncoderDecoder):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.encoder_head = HeadGaussianDiagonal(self.encoder_base.last_hidden_dim, config.model.latent_dim)

    def encode(self, x):
        x = self.encoder_base(x)
        z, mu, logvar = self.encoder_head(x)
        return z, mu, logvar

    def forward(self, original: torch.Tensor, proj_target: torch.Tensor, mode: str = "full"):
        x = self.encoder_base(original)
        z, mu, logvar = self.encoder_head(x)
        recon = self.decoder(z)
        return recon, z, mu, logvar


class VAEFull(EncoderDecoder):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.encoder_head = HeadGaussianFull(self.encoder_base.last_hidden_dim, config.model.latent_dim)

    def encode(self, x):
        x = self.encoder_base(x)
        z, mu, L = self.encoder_head(x)
        return z, mu, L

    def forward(self, original: torch.Tensor, proj_target: torch.Tensor, mode: str = "full"):
        x = self.encoder_base(original)
        z, mu, L = self.encoder_head(x)
        recon = self.decoder(z)
        return recon, z, mu, L


class VAEIsotropic(EncoderDecoder):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.encoder_head = HeadGaussianIsotropic(self.encoder_base.last_hidden_dim, config.model.latent_dim)

    def encode(self, x):
        x = self.encoder_base(x)
        z, mu, logvar = self.encoder_head(x)
        return z, mu, logvar

    def forward(self, original: torch.Tensor, proj_target: torch.Tensor, mode: str = "full"):
        x = self.encoder_base(original)
        z, mu, logvar = self.encoder_head(x)
        recon = self.decoder(z)
        return recon, z, mu, logvar


MODEL_LOOKUP = {"ae": AE, "vae": VAE, "vae-full": VAEFull, "vae-isotropic": VAEIsotropic, "pr": ProjectorReconstructor}


def create_model(config: Config) -> EncoderDecoder:
    model_type = config.model.type.lower()
    if model_type not in MODEL_LOOKUP:
        raise ValueError(f"Unknown model_type '{config.model.type}'. Expected one of: {list(MODEL_LOOKUP.keys())}")
    model_class = MODEL_LOOKUP[model_type]
    return model_class(config)


def save_model(model: EncoderDecoder, dir_path: str = "models"):
    os.makedirs(dir_path, exist_ok=True)
    model.config.save_to_yaml(dir_path)
    filename = model.config.create_filename()
    filepath = os.path.join(dir_path, f"{filename}.pt")
    torch.save(model.state_dict(), filepath)


def load_model(model_name: str) -> EncoderDecoder:
    config = Config.load_from_yaml(f"./models/{model_name}.yaml")
    model = create_model(config)
    state_dict = torch.load(f"./models/{model_name}.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model
