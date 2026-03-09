from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from config import Config
from loss_function_impl import (
    differential_entropy,
    differential_entropy_cholesky,
    jacobian_frobenius_hutchinson,
    kl_divergence,
    kl_divergence_cholesky,
)


class ReconstructionLoss(ABC):
    @abstractmethod
    def forward(
        self,
        original: torch.Tensor,
        reconstruction: torch.Tensor,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def __call__(self, original: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        pass


class ReconstructionLossBCE(nn.Module, ReconstructionLoss):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(
        self,
        original: torch.Tensor,
        reconstruction: torch.Tensor,
    ) -> torch.Tensor:
        return self.bce(reconstruction, original)


class ReconstructionLossMSE(nn.Module, ReconstructionLoss):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        original: torch.Tensor,
        reconstruction: torch.Tensor,
    ) -> torch.Tensor:
        return self.mse(reconstruction, original)


class ReconstructionLossZero(nn.Module, ReconstructionLoss):
    zero: torch.Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer("zero", torch.tensor(0.0))

    def forward(
        self,
        original: torch.Tensor,
        reconstruction: torch.Tensor,
    ) -> torch.Tensor:
        return self.zero


class ProjectionLossMSELatent(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        original: torch.Tensor,
        proj_target: torch.Tensor,
        reconstruction: torch.Tensor,
        latent: torch.Tensor,
        latent_mean: torch.Tensor,
        latent_covar: torch.Tensor,
    ) -> torch.Tensor:
        return self.mse(latent, proj_target)


class ProjectionLossMSELatentMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        original: torch.Tensor,
        proj_target: torch.Tensor,
        reconstruction: torch.Tensor,
        latent: torch.Tensor,
        latent_mean: torch.Tensor,
        latent_covar: torch.Tensor,
    ) -> torch.Tensor:
        return self.mse(latent_mean, proj_target)


class RegularizationLossKLDiv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        original: torch.Tensor,
        proj_target: torch.Tensor,
        reconstruction: torch.Tensor,
        latent: torch.Tensor,
        latent_mean: torch.Tensor,
        latent_covar: torch.Tensor,
    ) -> torch.Tensor:
        return kl_divergence(latent_mean, latent_covar)


class RegularizationLossKLDivCholesky(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        original: torch.Tensor,
        proj_target: torch.Tensor,
        reconstruction: torch.Tensor,
        latent: torch.Tensor,
        latent_mean: torch.Tensor,
        latent_covar: torch.Tensor,
    ) -> torch.Tensor:
        return kl_divergence_cholesky(latent_mean, latent_covar)


class RegularizationLossDE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        original: torch.Tensor,
        proj_target: torch.Tensor,
        reconstruction: torch.Tensor,
        latent: torch.Tensor,
        latent_mean: torch.Tensor,
        latent_covar: torch.Tensor,
    ) -> torch.Tensor:
        return differential_entropy(latent_mean, latent_covar)


class RegularizationLossDECholesky(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        original: torch.Tensor,
        proj_target: torch.Tensor,
        reconstruction: torch.Tensor,
        latent: torch.Tensor,
        latent_mean: torch.Tensor,
        latent_covar: torch.Tensor,
    ) -> torch.Tensor:
        return differential_entropy_cholesky(latent_mean, latent_covar)


class RegularizationLossJacobianFrobenius(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        original: torch.Tensor,
        proj_target: torch.Tensor,
        reconstruction: torch.Tensor,
        latent: torch.Tensor,
        latent_mean: torch.Tensor,
        latent_covar: torch.Tensor,
    ) -> torch.Tensor:
        return jacobian_frobenius_hutchinson(reconstruction, latent)


class ZeroLoss(nn.Module):
    zero: torch.Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer("zero", torch.tensor(0.0))

    def forward(
        self,
        original: torch.Tensor,
        proj_target: torch.Tensor,
        reconstruction: torch.Tensor,
        latent: torch.Tensor,
        latent_mean: torch.Tensor,
        latent_covar: torch.Tensor,
    ) -> torch.Tensor:
        return self.zero


class LossFn(nn.Module):
    def __init__(
        self,
        loss_recon: ReconstructionLoss,
        loss_proj: nn.Module,
        loss_reg: nn.Module,
        weight_recon: float = 1.0,
        weight_proj: float = 0.0,
        weight_reg: float = 0.0,
    ):
        super().__init__()
        self.loss_recon = loss_recon
        self.loss_proj = loss_proj
        self.loss_reg = loss_reg
        self.weight_recon = weight_recon
        self.weight_proj = weight_proj
        self.weight_reg = weight_reg

    def forward(
        self,
        original: torch.Tensor,
        proj_target: torch.Tensor,
        reconstruction: torch.Tensor,
        latent: torch.Tensor,
        latent_mean: torch.Tensor,
        latent_covar: torch.Tensor,
    ):
        recon = self.loss_recon(original, reconstruction)
        proj = self.loss_proj(original, proj_target, reconstruction, latent, latent_mean, latent_covar)
        reg = self.loss_reg(original, proj_target, reconstruction, latent, latent_mean, latent_covar)
        total = self.weight_recon * recon + self.weight_proj * proj + self.weight_reg * reg
        return total, recon, proj, reg


def create_loss_function(config: Config) -> LossFn:
    # Reconstruction loss
    if config.loss_recon is None:
        loss_recon = ReconstructionLossZero()
        weight_recon = 0.0
    else:
        if config.loss_recon.loss_fn.lower() == "bce":
            loss_recon = ReconstructionLossBCE()
        elif config.loss_recon.loss_fn.lower() == "mse":
            loss_recon = ReconstructionLossMSE()
        else:
            raise ValueError(f"Unknown reconstruction loss: {config.loss_recon.loss_fn}")
        weight_recon = config.loss_recon.weight

    # Projection loss
    if config.loss_proj is None:
        loss_proj = ZeroLoss()
        weight_proj = 0.0
    else:
        target = config.loss_proj.target.lower()
        if config.loss_proj.loss_fn.lower() == "mse":
            if target == "latent":
                loss_proj = ProjectionLossMSELatent()
            elif target == "latent_mean":
                loss_proj = ProjectionLossMSELatentMean()
            else:
                raise ValueError(f"Unknown projection target: {target}")
        else:
            raise ValueError(f"Unknown projection loss: {config.loss_proj.loss_fn}")
        weight_proj = config.loss_proj.weight

    # Regularization loss
    if config.loss_reg is None:
        loss_reg = ZeroLoss()
        weight_reg = 0.0
    else:
        # Pick from your regularization classes
        if config.loss_reg.loss_fn.lower() == "kl_div":
            loss_reg = RegularizationLossKLDiv()
        elif config.loss_reg.loss_fn.lower() == "kl_div_cholesky":
            loss_reg = RegularizationLossKLDivCholesky()
        elif config.loss_reg.loss_fn.lower() == "de":
            loss_reg = RegularizationLossDE()
        elif config.loss_reg.loss_fn.lower() == "de_cholesky":
            loss_reg = RegularizationLossDECholesky()
        elif config.loss_reg.loss_fn.lower() == "jacobian_frobenius":
            loss_reg = RegularizationLossJacobianFrobenius()
        else:
            raise ValueError(f"Unknown regularization loss: {config.loss_reg.loss_fn}")
        weight_reg = config.loss_reg.weight

    # Combine into a single loss module
    loss_fn = LossFn(
        loss_recon=loss_recon,
        loss_proj=loss_proj,
        loss_reg=loss_reg,
        weight_recon=weight_recon,
        weight_proj=weight_proj,
        weight_reg=weight_reg,
    )

    return loss_fn


def create_reconstruction_loss(config: Config) -> LossFn:
    # Reconstruction loss
    if config.loss_recon is None:
        loss_recon = ReconstructionLossZero()
        weight_recon = 0.0
    else:
        if config.loss_recon.loss_fn.lower() == "bce":
            loss_recon = ReconstructionLossBCE()
        elif config.loss_recon.loss_fn.lower() == "mse":
            loss_recon = ReconstructionLossMSE()
        else:
            raise ValueError(f"Unknown reconstruction loss: {config.loss_recon.loss_fn}")
        weight_recon = config.loss_recon.weight

    loss_proj = ZeroLoss()
    weight_proj = 0.0

    loss_reg = ZeroLoss()
    weight_reg = 0.0

    return LossFn(
        loss_recon=loss_recon,
        loss_proj=loss_proj,
        loss_reg=loss_reg,
        weight_recon=weight_recon,
        weight_proj=weight_proj,
        weight_reg=weight_reg,
    )


def create_projection_loss(config: Config) -> LossFn:
    loss_recon = ReconstructionLossZero()
    weight_recon = 0.0

    # Projection loss
    if config.loss_proj is None:
        loss_proj = ZeroLoss()
        weight_proj = 0.0
    else:
        target = config.loss_proj.target.lower()
        if config.loss_proj.loss_fn.lower() == "mse":
            if target == "latent":
                loss_proj = ProjectionLossMSELatent()
            elif target == "latent_mean":
                loss_proj = ProjectionLossMSELatentMean()
            else:
                raise ValueError(f"Unknown projection target: {target}")
        else:
            raise ValueError(f"Unknown projection loss: {config.loss_proj.loss_fn}")
        weight_proj = config.loss_proj.weight

    loss_reg = ZeroLoss()
    weight_reg = 0.0

    return LossFn(
        loss_recon=loss_recon,
        loss_proj=loss_proj,
        loss_reg=loss_reg,
        weight_recon=weight_recon,
        weight_proj=weight_proj,
        weight_reg=weight_reg,
    )
