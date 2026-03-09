import math

import torch


@torch.jit.script
def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()


@torch.jit.script
def kl_divergence_cholesky(mu: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    diag = torch.diagonal(L, dim1=1, dim2=2)
    log_det = 2 * torch.sum(torch.log(torch.clamp(diag, min=1e-8)), dim=1)
    trace = torch.sum(L**2, dim=(1, 2))
    mu_term = torch.sum(mu**2, dim=1)
    k = L.size(1)
    kl = 0.5 * (trace + mu_term - k - log_det)
    return kl.mean()


@torch.jit.script
def differential_entropy(_, logvar):
    constant = torch.log(torch.tensor(2 * math.pi, device=logvar.device))
    ent = 0.5 * torch.sum(1 + logvar + constant, dim=1)
    return ent.mean()


@torch.jit.script
def differential_entropy_cholesky(_, L):
    latent_dim = L.size(1)
    diag_L = torch.diagonal(L, dim1=1, dim2=2)
    log_det_cov = 2.0 * torch.sum(torch.log(diag_L + 1e-8), dim=1)
    constant = 0.5 * latent_dim * (1 + torch.log(torch.tensor(2 * math.pi, device=L.device)))
    entropy = constant + 0.5 * log_det_cov
    return entropy.mean()


def jacobian_frobenius_hutchinson(recon_x: torch.Tensor, z: torch.Tensor, num_probes: int = 4) -> torch.Tensor:
    # NOTE: assumes no cross-sample ops (e.g. BatchNorm) in the decoder,
    # otherwise per-sample gradients from autograd.grad are incorrect.
    assert recon_x.requires_grad, "recon_x must require grad for Jacobian estimation."
    assert z.requires_grad, "z must require grad for Jacobian estimation."
    B, D = recon_x.shape
    r = torch.randn(num_probes, B, D, device=recon_x.device, dtype=recon_x.dtype)
    total = torch.tensor(0.0, device=recon_x.device)
    for i in range(num_probes):
        s = (recon_x * r[i]).sum(dim=1)
        Jt_r = torch.autograd.grad(
            outputs=s,
            inputs=z,
            grad_outputs=torch.ones_like(s),
            retain_graph=True,
            create_graph=True,
        )[0]
        total += (Jt_r.pow(2).sum(dim=1)).sum()
    return total / (num_probes * B)
