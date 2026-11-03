"""Williams-Otto reactor with cooling-water CaCO3 fouling — Python port.

This package is a faithful reimplementation of the MATLAB reference at
https://github.com/joaoplcoutinho/wo_fouling_simulator — the Williams-Otto
reactor with cooling-water CaCO3 fouling kinetics from Souza & Costa (2019).

Authorship and origin: Joel Sansana and João Coutinho (CIEPQPF, Coimbra, 2021).
Original .m files joint-authored with José Granjo (CERENA, IST, ULisboa).
See docs/AUDIT.md for the architecture, validation contract, and build order.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .parameters import WoParameters, default_parameters
from .kinetics import log_rate_constants, mass_rates, reaction_heat
from .reactor import OpenLoopReactor

__all__ = [
    "WoParameters",
    "default_parameters",
    "log_rate_constants",
    "mass_rates",
    "reaction_heat",
    "OpenLoopReactor",
    "__version__",
]
