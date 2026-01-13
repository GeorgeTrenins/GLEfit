from ._base import BaseEmbedder
from .prony import PronyEmbedder
from .pronycos import PronyCosineEmbedder
from .gen2x2 import TwoAuxEmbedder
from .multi import MultiEmbedder

EMBEDDER_MAP = {
    "gen2x2": TwoAuxEmbedder,
    "multi": MultiEmbedder,
    "prony": PronyEmbedder,
    "pronycos": PronyCosineEmbedder,
}