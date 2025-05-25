# The implementation of complexity measures is from https://github.com/nitarshan/robust-generalization-measures
# It is based on the paper by Dziugaite et el. (2020) "In Search of Robust Measures of Generalization"
# https://arxiv.org/abs/2010.11924
import json

from typing import Dict, List, NamedTuple, Optional, Tuple
from enum import Enum, IntEnum
from copy import deepcopy
from torch import Tensor
from contextlib import contextmanager
import math

import os
import numpy as np
import torch
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
import torchvision
import torchvision.transforms as transforms
import argparse


class ComplexityType(Enum):
    # Measures from Fantastic Generalization Measures
    PARAMS = 20
    MARGIN = 21
    INVERSE_MARGIN = 22
    LOG_SPEC_INIT_MAIN = 29
    LOG_SPEC_ORIG_MAIN = 30
    LOG_PROD_OF_SPEC_OVER_MARGIN = 31
    LOG_PROD_OF_SPEC = 32
    FRO_OVER_SPEC = 33
    LOG_SUM_OF_SPEC_OVER_MARGIN = 34
    LOG_SUM_OF_SPEC = 35
    LOG_PROD_OF_FRO_OVER_MARGIN = 36
    LOG_PROD_OF_FRO = 37
    LOG_SUM_OF_FRO_OVER_MARGIN = 38
    LOG_SUM_OF_FRO = 39
    FRO_DIST = 40
    DIST_SPEC_INIT = 41
    PARAM_NORM = 42
    PATH_NORM_OVER_MARGIN = 43
    PATH_NORM = 44
    PACBAYES_INIT = 48
    PACBAYES_ORIG = 49
    PACBAYES_FLATNESS = 53
    PACBAYES_MAG_INIT = 56
    PACBAYES_MAG_ORIG = 57
    PACBAYES_MAG_FLATNESS = 61
    # Other Measures
    L2 = 100
    L2_DIST = 101

    @classmethod
    def data_dependent_measures(cls):
        return {
            cls.PACBAYES_ORIG,
            cls.PACBAYES_INIT,
            cls.PACBAYES_MAG_ORIG,
            cls.PACBAYES_MAG_INIT,
            cls.PACBAYES_FLATNESS,
            cls.PACBAYES_MAG_FLATNESS,
            cls.INVERSE_MARGIN,
            cls.LOG_PROD_OF_FRO_OVER_MARGIN,
            cls.LOG_SUM_OF_FRO_OVER_MARGIN,
            cls.LOG_PROD_OF_SPEC_OVER_MARGIN,
            cls.LOG_SUM_OF_SPEC_OVER_MARGIN,
            cls.PATH_NORM_OVER_MARGIN,
        }

    @classmethod
    def acc_dependent_measures(cls):
        return {
            cls.PACBAYES_ORIG,
            cls.PACBAYES_INIT,
            cls.PACBAYES_MAG_ORIG,
            cls.PACBAYES_MAG_INIT,
            cls.PACBAYES_FLATNESS,
            cls.PACBAYES_MAG_FLATNESS,
        }


class EvaluationMetrics(NamedTuple):
    acc: float
    avg_loss: float
    num_correct: int
    num_to_evaluate_on: int
    all_complexities: Dict[ComplexityType, float]


# instance
CT = ComplexityType


# https://github.com/bneyshabur/generalization-bounds/blob/master/measures.py
@torch.no_grad()
def _reparam(model):
    def in_place_reparam(model, prev_layer=None):
        for child in model.children():
            # print("1", child, prev_layer)
            prev_layer = in_place_reparam(child, prev_layer)
            # print("2", child, prev_layer)
            if child._get_name() == 'Conv2d':
                prev_layer = child
                # print("this is conv", child, prev_layer)
            elif child._get_name() == 'BatchNorm2d':
                # print("this is batch", child, prev_layer)
                scale = child.weight / ((child.running_var + child.eps).sqrt())
                # prev_layer.bias.copy_(child.bias + (scale * (prev_layer.bias - child.running_mean)))
                perm = list(reversed(range(prev_layer.weight.dim())))
                prev_layer.weight.copy_((prev_layer.weight.permute(perm) * scale).permute(perm))
                child.bias.fill_(0)
                child.weight.fill_(1)
                child.running_mean.fill_(0)
                child.running_var.fill_(1)
        return prev_layer

    model = deepcopy(model)
    in_place_reparam(model)
    return model


def rescale_model(model):
    print(model)
    model = deepcopy(model)
    for name, param in model.named_parameters():
        if 'layer1.0.conv1' in name:
            param.data = param.data * 0.5
        elif 'layer1.0.conv2' in name:
            param.data = param.data * 2

    return model


@torch.no_grad()
def get_all_measures(model, init_model, dataloader) -> Dict[CT, float]:
    measures = {}

    # Reparameterize models
    model = _reparam(model)
    init_model = _reparam(init_model)

    device = next(model.parameters()).device
    m = len(dataloader.dataset)

    # Helper functions
    def get_weights_only(model) -> List[Tensor]:
        blacklist = {'bias', 'bn'}
        return [p for name, p in model.named_parameters() if all(x not in name for x in blacklist)]

    def get_vec_params(weights: List[Tensor]) -> Tensor:
        return torch.cat([p.view(-1) for p in weights], dim=0)

    def get_reshaped_weights(weights: List[Tensor]) -> List[Tensor]:
        return [p.view(p.shape[0], -1) for p in weights]

    # Model weights and initialization distance
    weights = get_weights_only(model)
    dist_init_weights = [p - q for p, q in zip(weights, get_weights_only(init_model))]
    w_vec = get_vec_params(weights)
    dist_w_vec = get_vec_params(dist_init_weights)
    num_params = len(w_vec)
    reshaped_weights = get_reshaped_weights(weights)
    dist_reshaped_weights = get_reshaped_weights(dist_init_weights)

    # VC-Dimension Based Measures
    measures[CT.PARAMS] = torch.tensor(num_params)  # Number of parameters
    print("PARAMS computation complete.")

    # Norm-based measures
    measures[CT.L2] = w_vec.norm(p=2)  # L2 norm
    measures[CT.L2_DIST] = dist_w_vec.norm(p=2)  # Distance in L2 space
    print("L2 and L2_DIST computation complete.")

    # Margin-based measures
    def _margin(model, dataloader: DataLoader) -> Tensor:
        margins = []
        for data, target in dataloader:
            data = data.to(device, dtype=torch.float)
            logits = model(data)
            correct_logit = logits[torch.arange(logits.shape[0]), target].clone()
            logits[torch.arange(logits.shape[0]), target] = float('-inf')
            max_other_logit = logits.data.max(1).values
            margin = correct_logit - max_other_logit
            margins.append(margin)
        return torch.cat(margins).kthvalue(m // 10)[0]

    margin = _margin(model, dataloader).abs()
    measures[CT.MARGIN] = 1 / margin
    measures[CT.INVERSE_MARGIN] = 1 / (margin ** 2)
    print("MARGIN and INVERSE_MARGIN computation complete.")

    # Spectral Norm and related measures
    fro_norms = torch.cat([p.norm('fro').unsqueeze(0) ** 2 for p in reshaped_weights])
    spec_norms = torch.cat([p.svd().S.max().unsqueeze(0) ** 2 for p in reshaped_weights])
    # Distance Frobenius Norms (Add this definition)
    dist_fro_norms = torch.cat([p.norm('fro').unsqueeze(0) ** 2 for p in dist_reshaped_weights])
    dist_spec_norms = torch.cat([p.svd().S.max().unsqueeze(0) ** 2 for p in dist_reshaped_weights])
    measures[CT.LOG_PROD_OF_SPEC] = spec_norms.log().sum()
    measures[CT.LOG_PROD_OF_SPEC_OVER_MARGIN] = measures[CT.LOG_PROD_OF_SPEC] - 2 * margin.log()
    print("Spectral norm computation complete.")

    # PAC-Bayesian measures
    measures[CT.PACBAYES_INIT] = (dist_fro_norms / fro_norms).sum()
    measures[CT.PACBAYES_FLATNESS] = (dist_fro_norms / (fro_norms + 1e-8)).sum().log()
    measures[CT.PACBAYES_MAG_INIT] = (fro_norms + dist_fro_norms).sum().log()
    measures[CT.PACBAYES_MAG_ORIG] = measures[CT.PACBAYES_MAG_INIT] - spec_norms.sum().log()
    measures[CT.PACBAYES_ORIG] = (dist_fro_norms / fro_norms).sum().log()  # New computation for PB-O
    print("PAC-Bayesian measures computation complete.")


    # Distance-based measures
    measures[CT.FRO_DIST] = dist_fro_norms.sum()
    measures[CT.DIST_SPEC_INIT] = dist_spec_norms.sum()
    print("Distance-based measures computation complete.")

    # Adjust for dataset size
    def adjust_measure(measure: CT, value: float) -> float:
        if measure.name.startswith('LOG_'):
            return 0.5 * (value - np.log(m))
        else:
            return np.sqrt(value)

    # Final adjustments and return
    return {k: adjust_measure(k, v.item()) for k, v in measures.items()}
