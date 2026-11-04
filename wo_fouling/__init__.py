"""Williams-Otto reactor with cooling-water CaCO3 fouling — Python port.

This package is a faithful reimplementation of the MATLAB reference at
https://github.com/joaoplcoutinho/wo_fouling_simulator — the Williams-Otto
reactor with cooling-water CaCO3 fouling kinetics from Souza & Costa (2019).

Authorship and origin: Joel Sansana and João Coutinho (CIEPQPF, Coimbra, 2021).
Original .m files joint-authored with José Granjo (CERENA, IST, ULisboa).
See PORT_NOTES.md for the source mapping, subsystem verdicts, and validation contract.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .parameters import WoParameters, default_parameters, open_loop_parameters, closed_loop_parameters
from .kinetics import log_rate_constants, mass_rates, reaction_heat
from .reactor import OpenLoopReactor
from .fouling import FoulingReactor, dynmodel_fouling
from .tube import TubeSimulator, tubo_fouling
from .steady_state import solve_open_loop_ss, solve_closed_loop_ss
from .controllers import DiscretePID, relay_autotune, tyreus_luyben_pid, amigo_pid
from .closed_loop import ClosedLoopSimulator, ClosedLoopConfig
from . import sim, kpi, clustering, ol_sysid, matio  # noqa: F401

__all__ = [
    "WoParameters",
    "default_parameters",
    "open_loop_parameters",
    "closed_loop_parameters",
    "log_rate_constants",
    "mass_rates",
    "reaction_heat",
    "OpenLoopReactor",
    "FoulingReactor",
    "dynmodel_fouling",
    "TubeSimulator",
    "tubo_fouling",
    "solve_open_loop_ss",
    "solve_closed_loop_ss",
    "DiscretePID",
    "relay_autotune",
    "tyreus_luyben_pid",
    "amigo_pid",
    "ClosedLoopSimulator",
    "ClosedLoopConfig",
    "__version__",
]
