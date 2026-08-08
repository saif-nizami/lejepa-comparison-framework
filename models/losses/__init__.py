from .nt_xent import NTXentLoss
from .negative_cosine import NegativeCosineSimilarity
from .vicreg import VICRegLoss
from .barlow_twins import BarlowTwinsLoss
from .sigreg import SIGRegLoss

__all__ = [
    "NTXentLoss",
    "NegativeCosineSimilarity",
    "VICRegLoss",
    "BarlowTwinsLoss",
    "SIGRegLoss",
]