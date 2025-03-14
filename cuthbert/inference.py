from typing import Callable, NamedTuple


# TODO: Add unified protocols for init/predict/update/smoother - this will need
# discussion on how to unify SMC inspired FeynmanKac with Kalman filtering/smoothing.
# Concretely these differ in the following ways:
# Dynamics: FeynmanKac has trajectory, Kalman has just the previous state
# Likelihood: FeynmancKac has trajectory, Kalman has state and observation


class Inference(NamedTuple):
    init: Callable
    predict: Callable
    update: Callable
    filter: Callable
    smoother: Callable
