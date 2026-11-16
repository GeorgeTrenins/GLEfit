from .debye import Debye
from .expohmic import ExpOhmic
from .brownian import BrownianOscillator

SPECTRAL_MAP = {
    "debye": Debye,
    "expohmic": ExpOhmic,
    "brownian": BrownianOscillator,
}
