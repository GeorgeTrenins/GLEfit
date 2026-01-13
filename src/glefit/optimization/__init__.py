from .NR import NewtonRaphson, EigenvectorFollowing
from ._base import Optimizer

OPT_MAP = {
    "NR": NewtonRaphson,
    "EF": EigenvectorFollowing
}