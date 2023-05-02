"""
gp1 allowed some practice on fitting gaussian processes which will be very useful. Here
we will begin experiments with acquisition functions
"""

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import torch

from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement

from botorch.optim import optimize_acqf


def obj(x):
    """
    our test objective function:
    f(x) = sin(x) + sin(10x/3)
    over domain (-6, 6) is multimodal and has global minimum at x=5.146
    """
    return np.sin(x) + np.sin((10./3.)*x)

def botorch_test():
    train_x = torch.rand(10, 1)*12 - 6
    train_obj = obj(train_x)

    best_val = train_obj.max()

    model = SingleTaskGP(train_X=train_x, train_Y=train_obj)

    EI = ExpectedImprovement(model=model, best_f = best_val)

    new_point_analytic, _ = optimize_acqf(
    acq_function=EI,
    bounds=torch.tensor([[-6.0], [6.0]]),
    q=1,
    num_restarts=20,
    raw_samples=100,
    options={},
    )
    print(new_point_analytic)


botorch_test()