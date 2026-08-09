"""
Factory for Self-Supervised Learning models.
"""

from __future__ import annotations

from models.ssl import (
    SimCLR,
    BYOL,
    VICReg,
    BarlowTwins,
    LeJEPA,
)


def build_model(config):

    name = config.model.name.lower()

    if name == "simclr":

        return SimCLR(
            projection_dim=config.model.projection_dim,
            hidden_dim=config.model.hidden_dim,
            temperature=config.simclr.temperature,
            pretrained_backbone=config.model.pretrained_backbone,
        )

    elif name == "byol":

        return BYOL(
            projection_dim=config.byol.projection_dim,
            hidden_dim=config.model.hidden_dim,
            predictor_hidden_dim=config.byol.predictor_hidden_dim,
            momentum=config.byol.momentum,
            pretrained_backbone=config.model.pretrained_backbone,
        )

    elif name == "vicreg":

        return VICReg(
            projection_dim=config.model.projection_dim,
            hidden_dim=config.model.hidden_dim,
            sim_coeff=config.vicreg.sim_coeff,
            std_coeff=config.vicreg.std_coeff,
            cov_coeff=config.vicreg.cov_coeff,
            pretrained_backbone=config.model.pretrained_backbone,
        )

    elif name == "barlow_twins":

        return BarlowTwins(
            projection_dim=config.model.projection_dim,
            hidden_dim=config.model.hidden_dim,
            lambd=config.barlow_twins.lambda_coeff,
            pretrained_backbone=config.model.pretrained_backbone,
        )

    elif name == "lejepa":

        return LeJEPA(
            projection_dim=config.model.projection_dim,
            hidden_dim=config.model.hidden_dim,
            lambda_sigreg=config.lejepa.lambda_sigreg,
            num_slices=config.lejepa.num_slices,
            epps_points=config.lejepa.epps_points,
            epps_tmax=config.lejepa.epps_tmax,
            pretrained_backbone=config.model.pretrained_backbone,
        )

    raise ValueError(
        f"Unknown SSL model '{config.model.name}'."
    )