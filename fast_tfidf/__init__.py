import importlib.metadata

from . import main


__version__ = importlib.metadata.version(__package__)
get_vocabulary_and_idf_weights = main.get_vocabulary_and_idf_weights

__all__ = ["get_vocabulary_and_idf_weights"]
