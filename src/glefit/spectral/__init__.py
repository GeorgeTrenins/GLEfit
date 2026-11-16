from .debye import Debye
from .expohmic import ExpOhmic

SPECTRAL_MAP = {
    "debye": Debye,
    "expohmic": ExpOhmic,
}
