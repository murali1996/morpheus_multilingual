__version__ = "1.0.0"
__author__ = 'Sai Muralidhar Jayanthi, Pratapa, Adithya'
__email__ = "vpratapa@andrew.cmu.edu"

from .morpheus_nmt import MorpheusFairseqTransformerNMT
from .utils.create_dictionary import get_candidates
from .utils.create_perturbations import create_inflections_lookup, get_sentences_meta
from .utils.create_reinflections import get_reinflections
from .utils.log import configure_logs


def get_version():
    return __version__


__all__ = [
    "get_version",
    "MorpheusFairseqTransformerNMT",
    "get_candidates",
    "get_reinflections",
    "create_inflections_lookup",
    "get_sentences_meta",
    "configure_logs"
]

configure_logs()
