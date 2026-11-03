"""map2loc: BEV-to-map localization with attitude-aware rectification.

This branch combines the updated perspective-camera / heavy-corruption
observation generator (``data_gen``) with BEV rectification, classical and
learned matching, and temporal filtering.
"""

__version__ = "0.2.0"