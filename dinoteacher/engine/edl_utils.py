"""
Beta-distribution uncertainty from classification scores (lightweight EDL proxy).
"""

import torch
import torch.nn.functional as F


def beta_uncertainty_from_score(scores, strength):
    """
    Compute Beta variance u from per-box confidence p and evidence strength lambda.

    alpha = p * lambda, beta = (1 - p) * lambda
    u = alpha * beta / ((alpha + beta + 1) * (alpha + beta)^2)
    """
    p = scores.float().clamp(1e-6, 1.0 - 1e-6)
    lam = float(strength)
    alpha = p * lam
    beta = (1.0 - p) * lam
    s = alpha + beta
    return alpha * beta / ((s + 1.0) * s * s)


def evidence_to_alpha_beta(scores, evidence_strength, min_lambda=1.0):
    """
    Map confidence scores and evidence strength to Beta(alpha, beta) parameters.
    """
    p = scores.float().clamp(1e-6, 1.0 - 1e-6)
    lam = torch.as_tensor(evidence_strength, dtype=p.dtype, device=p.device).clamp(
        min=float(min_lambda)
    )
    alpha = p * lam
    beta = (1.0 - p) * lam
    return alpha, beta


def uncertainty_from_alpha_beta(alpha, beta):
    """
    Eq.8 style Beta variance uncertainty:
    u = alpha * beta / ((alpha + beta + 1) * (alpha + beta)^2)
    """
    alpha = alpha.float().clamp(min=1e-6)
    beta = beta.float().clamp(min=1e-6)
    s = alpha + beta
    return alpha * beta / ((s + 1.0) * s * s)


def eq11_mask(scores, uncertainty, conf_threshold, unc_threshold):
    """
    Eq.11 pseudo-label gate:
    g = 1[p > tau_p] * 1[u < tau_u]
    """
    return (scores > conf_threshold) & (uncertainty < unc_threshold)


def edl_cls_loss_from_alpha_beta(alpha, beta, target):
    """
    Eq.7 binary EDL loss:
    L = -y*log(alpha/(alpha+beta)) - (1-y)*log(beta/(alpha+beta))
    """
    alpha = alpha.float().clamp(min=1e-6)
    beta = beta.float().clamp(min=1e-6)
    target = target.float()
    s = alpha + beta
    p_pos = (alpha / s).clamp(min=1e-6, max=1.0 - 1e-6)
    p_neg = (beta / s).clamp(min=1e-6, max=1.0 - 1e-6)
    return -(target * torch.log(p_pos) + (1.0 - target) * torch.log(p_neg))


def edl_regu_loss(per_sample_cls_loss, evidence_strength):
    """
    Eq.10 regularizer:
    L_regu = mean((l_cls - 1/lambda)^2)
    """
    evidence_strength = evidence_strength.float().clamp(min=1e-6)
    return torch.mean((per_sample_cls_loss - 1.0 / evidence_strength) ** 2)


def per_sample_ce_loss(logits, targets):
    """
    Helper for regularizer: unreduced classification loss per ROI.
    """
    return F.cross_entropy(logits, targets, reduction="none")
