import os
import zlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import yaml

ACTIVATION_LOOKUP = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "leakyrelu": nn.LeakyReLU,
    "softmax": nn.Softmax,
    "selu": nn.SELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "elu": nn.ELU,
    "hardtanh": nn.Hardtanh,
}


def get_activation_func(act) -> nn.Module:
    return ACTIVATION_LOOKUP[act]() if act in ACTIVATION_LOOKUP else nn.Identity()


@dataclass
class LayerConfig(ABC):
    @abstractmethod
    def to_module(self, in_shape: int | Tuple[int, int, int]) -> Tuple[nn.Module, int | Tuple[int, int, int]]:
        pass


@dataclass
class LinearConfig(LayerConfig):
    out_features: int
    type: str = "linear"
    activation: Optional[str] = None

    def to_module(self, in_features: int) -> Tuple[nn.Module, int]:
        return (
            nn.Sequential(nn.Linear(in_features, self.out_features), get_activation_func(self.activation)),
            self.out_features,
        )


@dataclass
class Conv2dConfig(LayerConfig):
    out_channels: int
    kernel_size: int
    type: str = "conv2d"
    stride: int = 1
    padding: int = 0
    activation: Optional[str] = None

    def to_module(self, in_shape: Tuple[int, int, int]) -> Tuple[nn.Module, Tuple[int, int, int]]:
        C_in, H_in, W_in = in_shape
        H_out = (H_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        module = nn.Sequential(
            nn.Conv2d(
                in_channels=C_in,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            get_activation_func(self.activation),
        )

        return module, (self.out_channels, H_out, W_out)


@dataclass
class ConvTranspose2dConfig(LayerConfig):
    out_channels: int
    kernel_size: int
    type: str = "convtranspose2d"
    stride: int = 1
    padding: int = 0
    output_padding: int = 0
    activation: Optional[str] = None

    def to_module(self, in_shape: Tuple[int, int, int]) -> Tuple[nn.Module, Tuple[int, int, int]]:
        C_in, H_in, W_in = in_shape

        # Compute output height and width for ConvTranspose2d
        H_out = (H_in - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        W_out = (W_in - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        module = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=C_in,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
            ),
            get_activation_func(self.activation),
        )
        return module, (self.out_channels, H_out, W_out)


@dataclass
class AdaptiveAvgPool2dConfig(LayerConfig):
    out_height: int
    out_width: int
    type: str = "adaptiveavgpool2d"

    def to_module(self, in_shape: Tuple[int, int, int]) -> Tuple[nn.Module, Tuple[int, int, int]]:
        return nn.AdaptiveAvgPool2d((self.out_height, self.out_width)), (
            in_shape[0],
            self.out_height,
            self.out_width,
        )


@dataclass
class MaxPool2dConfig(LayerConfig):
    kernel_size: int
    type: str = "maxpool2d"
    stride: Optional[int] = None  # defaults to kernel_size if None

    def to_module(self, in_shape: Tuple[int, int, int]) -> Tuple[nn.Module, Tuple[int, int, int]]:
        C, H, W = in_shape
        stride = self.stride if self.stride is not None else self.kernel_size
        H_out = (H - self.kernel_size) // stride + 1
        W_out = (W - self.kernel_size) // stride + 1
        return nn.MaxPool2d(kernel_size=self.kernel_size, stride=stride), (C, H_out, W_out)


@dataclass
class BatchNorm1dConfig(LayerConfig):
    type: str = "batchnorm1d"

    def to_module(self, in_features: int) -> Tuple[nn.Module, int]:
        return nn.BatchNorm1d(in_features), in_features


@dataclass
class BatchNorm2dConfig(LayerConfig):
    num_features: int
    type: str = "batchnorm2d"

    def to_module(self, in_shape: Tuple[int, int, int]) -> Tuple[nn.Module, Tuple[int, int, int]]:
        C, _, _ = in_shape
        if self.num_features != C:
            raise ValueError(f"BatchNorm2d num_features ({self.num_features}) does not match input channels ({C})")
        return nn.BatchNorm2d(self.num_features), in_shape


@dataclass
class Dropout1dConfig(LayerConfig):
    p: float = 0.5
    type: str = "dropout1d"

    def to_module(self, in_features: int) -> Tuple[nn.Module, int]:
        return nn.Dropout1d(self.p), in_features


@dataclass
class FlattenConfig(LayerConfig):
    type: str = "flatten"

    def to_module(self, in_shape: Tuple[int, int, int]) -> Tuple[nn.Module, int]:
        # Compute total number of features when flattened
        out_features = int(torch.tensor(in_shape).prod().item())
        return nn.Flatten(), out_features


@dataclass
class UnflattenConfig(LayerConfig):
    out_channels: int
    out_height: int
    out_width: int
    type: str = "unflatten"

    def to_module(self, in_features: int) -> Tuple[nn.Module, Tuple[int, int, int]]:
        class Unflatten(nn.Module):
            def __init__(self, c: int, h: int, w: int):
                super().__init__()
                self.c, self.h, self.w = c, h, w

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.view(x.size(0), self.c, self.h, self.w)

        return Unflatten(self.out_channels, self.out_height, self.out_width), (
            self.out_channels,
            self.out_height,
            self.out_width,
        )


LAYER_TYPE_MAP = {
    "adaptiveavgpool2d": AdaptiveAvgPool2dConfig,
    "batchnorm1d": BatchNorm1dConfig,
    "batchnorm2d": BatchNorm2dConfig,
    "conv2d": Conv2dConfig,
    "convtranspose2d": ConvTranspose2dConfig,
    "dropout1d": Dropout1dConfig,
    "flatten": FlattenConfig,
    "maxpool2d": MaxPool2dConfig,
    "linear": LinearConfig,
    "unflatten": UnflattenConfig,
}


@dataclass
class LossConfig:
    loss_fn: str  # Name of the loss function, e.g., "bce", "mse"
    weight: float  # Scaling factor for the loss term

    def __str__(self):
        return f"LossConfig(\n    loss_fn='{self.loss_fn}',\n    weight={self.weight}  )"


@dataclass
class ProjectionLossConfig(LossConfig):
    target: str  # The variable under consideration, e.g., "latent", "latent_mean"

    def __str__(self):
        return (
            f"ProjectionLossConfig(\n"
            f"    target='{self.target}',\n"
            f"    loss_fn='{self.loss_fn}',\n"
            f"    weight={self.weight}\n"
            f"  )"
        )


@dataclass
class ModelConfig:
    type: Literal["ae", "pr", "vae"]
    io_dim: int
    latent_dim: int
    encoder_layers: List[LayerConfig]
    decoder_layers: List[LayerConfig]

    def __str__(self):
        return (
            f"ModelConfig(\n"
            f"    type='{self.type}',\n"
            f"    io_dim={self.io_dim},\n"
            f"    latent_dim={self.latent_dim},\n"
            f"    encoder_layers={self.encoder_layers},\n"
            f"    decoder_layers={self.decoder_layers}\n"
            f"  )"
        )


@dataclass
class TrainingConfig:
    max_epochs: int
    batch_size: int
    learning_rate: float
    patience: int
    seed: int

    def __str__(self):
        return (
            f"TrainingConfig(\n"
            f"    max_epochs={self.max_epochs},\n"
            f"    batch_size={self.batch_size},\n"
            f"    learning_rate={self.learning_rate},\n"
            f"    patience={self.patience},\n"
            f"    seed={self.seed}\n"
            f"  )"
        )


@dataclass
class Config:
    dataset: str  # Dataset name, e.g., "mnist"
    projection: str  # Projection type, e.g., "tsne"
    comment: str
    model: ModelConfig
    loss_recon: LossConfig
    loss_proj: Optional[ProjectionLossConfig]
    loss_reg: Optional[LossConfig]
    training: TrainingConfig

    def create_filename(self) -> str:
        base = f"{self.dataset}_{self.projection}_{self.model.type}_{self.comment}"
        hash_int = zlib.crc32(self.__str__().encode())
        hash_str = f"{hash_int:08x}"
        return f"{base}_{hash_str}"

    def save_to_yaml(self, dir_path: str) -> str:
        config_dict = asdict(self)
        filename = self.create_filename()
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, f"{filename}.yaml")
        with open(file_path, "w") as file:
            yaml.dump(config_dict, file, default_flow_style=False)
        return file_path

    @classmethod
    def parse_layer(cls, cfg: dict) -> LayerConfig:
        layer_type = cfg["type"].lower()
        if layer_type not in LAYER_TYPE_MAP:
            raise ValueError(f"Unknown layer type: {cfg['type']}")
        return LAYER_TYPE_MAP[layer_type](**cfg)

    @classmethod
    def load_from_yaml(cls, file_path: str):
        with open(file_path, "r") as file:
            config_dict = yaml.safe_load(file)

        model_dict = config_dict.pop("model")

        encoder_layers = [Config.parse_layer(lc) for lc in model_dict.pop("encoder_layers")]
        decoder_layers = [Config.parse_layer(lc) for lc in model_dict.pop("decoder_layers")]

        model_cfg = ModelConfig(
            type=model_dict["type"],
            io_dim=model_dict["io_dim"],
            latent_dim=model_dict["latent_dim"],
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
        )

        training_cfg = TrainingConfig(**config_dict.pop("training"))
        loss_recon_cfg = LossConfig(**config_dict.pop("loss_recon"))

        # Load projection loss if it exists
        loss_proj_cfg = config_dict.pop("loss_proj")
        if loss_proj_cfg is not None:
            loss_proj_cfg = ProjectionLossConfig(**loss_proj_cfg)

        # Load regularization loss if it exists
        loss_reg_cfg = config_dict.pop("loss_reg")
        if loss_reg_cfg is not None:
            loss_reg_cfg = LossConfig(**loss_reg_cfg)

        return Config(
            model=model_cfg,
            training=training_cfg,
            loss_recon=loss_recon_cfg,
            loss_proj=loss_proj_cfg,
            loss_reg=loss_reg_cfg,
            **config_dict,
        )

    def __str__(self):
        return (
            f"Config(\n"
            f"  dataset='{self.dataset}',\n"
            f"  projection='{self.projection}',\n"
            f"  comment='{self.comment}',\n"
            f"  model={self.model},\n"
            f"  loss_recon={self.loss_recon},\n"
            f"  loss_proj={self.loss_proj},\n"
            f"  loss_reg={self.loss_reg},\n"
            f"  training={self.training}\n"
            f")"
        )


def test() -> None:
    config = Config(
        dataset="mnist",
        projection="tsne",
        comment="test",
        model=ModelConfig(
            type="ae",
            io_dim=784,
            latent_dim=2,
            encoder_layers=[
                UnflattenConfig(out_channels=1, out_height=28, out_width=28),
                Conv2dConfig(out_channels=32, kernel_size=3, stride=2, padding=1, activation="relu"),
                BatchNorm2dConfig(num_features=32),
                Conv2dConfig(out_channels=64, kernel_size=3, stride=2, padding=1, activation="relu"),
                BatchNorm2dConfig(num_features=64),
                Conv2dConfig(out_channels=128, kernel_size=3, stride=2, padding=1, activation="relu"),
                BatchNorm2dConfig(num_features=128),
                FlattenConfig(),
            ],
            decoder_layers=[
                LinearConfig(out_features=128 * 4 * 4),
                UnflattenConfig(out_channels=128, out_height=4, out_width=4),
                ConvTranspose2dConfig(
                    out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1, activation="relu"
                ),
                BatchNorm2dConfig(num_features=64),
                ConvTranspose2dConfig(
                    out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1, activation="relu"
                ),
                BatchNorm2dConfig(num_features=32),
                ConvTranspose2dConfig(
                    out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1, activation="sigmoid"
                ),
                AdaptiveAvgPool2dConfig(out_height=28, out_width=28),
                FlattenConfig(),
            ],
        ),
        loss_recon=LossConfig(loss_fn="bce", weight=1.0),
        loss_proj=ProjectionLossConfig(target="latent_mean", loss_fn="mse", weight=0.5),
        loss_reg=LossConfig(loss_fn="kl_div", weight=0.01),
        training=TrainingConfig(max_epochs=100, batch_size=64, learning_rate=1e-3, patience=5, seed=777),
    )

    print("Original config:")
    print(config)

    directory = "."
    file_path = config.save_to_yaml(directory)
    loaded_config = Config.load_from_yaml(file_path)
    print("Loaded config:")
    print(loaded_config)


if __name__ == "__main__":
    test()
