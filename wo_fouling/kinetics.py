"""Williams-Otto reaction kinetics — Arrhenius rate helpers.

This module is the pure-math layer that produces the three rate constants and
the resulting mass-basis reaction rates for the WO reactor:

    A + B -> C           r1 = k1(T) * xa * xb * MR
    C + B -> E + P       r2 = k2(T) * xb * xc * MR
    C + P -> G           r3 = k3(T) * xc * xp * MR

The mass-basis rate ``rjmr`` (kg/s of reaction progress) is what the material
balances consume. The convention matches ``dynmodel.m`` and ``dynmodel_fouling.m``
in the upstream MATLAB reference bit-for-bit (see PORT_NOTES.md §2 and §3).

Note on units:
    ``Ea1..Ea3`` in the upstream ``modelparameters.m`` are in **K** (not J/mol),
    and ``Rgas`` is **1.987204... cal/(K mol)**, so the dimensionless
    ``Ea / (Rgas * T)`` group that appears in the upstream code is already
    dimensionless as the K and cal/(K mol) cancel. We follow the upstream
    convention verbatim — do not "fix" the units (see PORT_NOTES.md §2).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .parameters import WoParameters


# -----------------------------------------------------------------------------
# Helpers (pure functions, no state)
# -----------------------------------------------------------------------------

def log_rate_constants(
    Tr: np.ndarray | float,
    par: WoParameters,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the three natural-log rate constants ``ln k_j(Tr)``.

    Direct port of the corresponding block in ``dynmodel.m``:

        Trinv = 1 / Tr;
        lnkl1 = log(A1) - Ea1 * Trinv;
        lnkl2 = log(A2) - Ea2 * Trinv;
        lnkl3 = log(A3) - Ea3 * Trinv;

    Parameters
    ----------
    Tr : float or ndarray
        Reactor temperature, K. Must be strictly positive (we floor at 1 K
        defensively — the upstream ``dynmodel.m`` floors state negatives to 0
        which would divide by zero; we use a tighter floor and rely on the
        state to keep Tr positive in normal operation).
    par : WoParameters
        Container of ``A1..A3`` and ``Ea1..Ea3``.

    Returns
    -------
    (lnkl1, lnkl2, lnkl3) : tuple of ndarray
        Same shape as ``Tr``. Values are the natural-log of the rate
        constants (not the rates themselves).
    """
    Tr_safe = np.maximum(np.asarray(Tr, dtype=np.float64), 1.0)
    Trinv = 1.0 / Tr_safe
    lnkl1 = np.log(par.A1) - par.Ea1 * Trinv
    lnkl2 = np.log(par.A2) - par.Ea2 * Trinv
    lnkl3 = np.log(par.A3) - par.Ea3 * Trinv
    return lnkl1, lnkl2, lnkl3


def mass_rates(
    xa: np.ndarray | float,
    xb: np.ndarray | float,
    xc: np.ndarray | float,
    xp: np.ndarray | float,
    MR: np.ndarray | float,
    Tr: np.ndarray | float,
    par: WoParameters,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the mass-basis reaction rates ``r1mr, r2mr, r3mr`` (kg/s).

    Direct port of the corresponding block in ``dynmodel.m``:

        r1mr = exp(lnkl1) * xa * xb * MR;
        r2mr = exp(lnkl2) * xb * xc * MR;
        r3mr = exp(lnkl3) * xc * xp * MR;

    Note the upstream convention: rates are in kg of reaction *progress*
    per second, scaled by ``MR`` (total reactor mass), not by ``VR``. This
    keeps the material balances dimensionally consistent without ever
    needing molar masses (the WO benchmark works in mass fractions).

    Parameters
    ----------
    xa, xb, xc, xp : float or ndarray
        Mass fractions of A, B, C, P in the reactor (dimensionless, [0,1]).
        They should sum to <= 1 together with ``xe``, ``xg``; we do not
        renormalise here.
    MR : float or ndarray
        Total reactor hold-up mass, kg. Must be strictly positive (we floor
        at ``1e-10`` matching the upstream safety check).
    Tr : float or ndarray
        Reactor temperature, K.
    par : WoParameters
        Kinetics container.

    Returns
    -------
    (r1mr, r2mr, r3mr) : tuple of ndarray
        Same shape as the broadcast of inputs.
    """
    MR_safe = np.maximum(np.asarray(MR, dtype=np.float64), 1e-10)
    lnkl1, lnkl2, lnkl3 = log_rate_constants(Tr, par)
    k1 = np.exp(lnkl1)
    k2 = np.exp(lnkl2)
    k3 = np.exp(lnkl3)
    r1mr = k1 * xa * xb * MR_safe
    r2mr = k2 * xb * xc * MR_safe
    r3mr = k3 * xc * xp * MR_safe
    return r1mr, r2mr, r3mr


# -----------------------------------------------------------------------------
# Convenience: reaction heat release (kW) at given state
# -----------------------------------------------------------------------------

def reaction_heat(
    r1mr: np.ndarray | float,
    r2mr: np.ndarray | float,
    r3mr: np.ndarray | float,
    par: WoParameters,
) -> np.ndarray | float:
    """Total rate of heat release by the three reactions (kW).

    Direct port of the corresponding block in ``dynmodel.m``:

        Qrmr = 2 * r1mr * (-DeltaH1) + 3 * r2mr * (-DeltaH2) + 1.5 * r3mr * (-DeltaH3);

    The stoichiometric multipliers (2, 3, 1.5) come from the per-reaction
    ``DeltaH`` references (per kg A consumed, per kg B consumed, per kg C
    consumed — see AUDIT §1.1). They are not the reaction stoichiometric
    coefficients themselves; they are the molar-mass ratios between the
    reference species and the heat-of-reaction basis (kJ/kg-X * kg-X/s = kJ/s
    = kW).

    Parameters
    ----------
    r1mr, r2mr, r3mr : float or ndarray
        Mass-basis reaction rates from :func:`mass_rates`.
    par : WoParameters
        Kinetics container (``DeltaH1..DeltaH3``).

    Returns
    -------
    Qrmr : float or ndarray
        Heat release rate, kW. Positive = heat generated.
    """
    r1mr = np.asarray(r1mr, dtype=np.float64)
    r2mr = np.asarray(r2mr, dtype=np.float64)
    r3mr = np.asarray(r3mr, dtype=np.float64)
    return (
        2.0 * r1mr * (-par.DeltaH1)
        + 3.0 * r2mr * (-par.DeltaH2)
        + 1.5 * r3mr * (-par.DeltaH3)
    )


# -----------------------------------------------------------------------------
# CLI sanity check
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    from .parameters import default_parameters

    par = default_parameters()
    Tr0 = 368.81  # upstream steady-state from sim_main.m
    lnkl1, lnkl2, lnkl3 = log_rate_constants(Tr0, par)
    print(f"At Tr = {Tr0} K:")
    print(f"  ln k1 = {lnkl1:.6f}  (k1 = {np.exp(lnkl1):.6e} s^-1 kg)")
    print(f"  ln k2 = {lnkl2:.6f}  (k2 = {np.exp(lnkl2):.6e} s^-1 kg)")
    print(f"  ln k3 = {lnkl3:.6f}  (k3 = {np.exp(lnkl3):.6e} s^-1 kg)")

    # At the sim_main.m steady state, the reactor holds massT=2105 kg with
    # mass fractions roughly xi = [0.37, 0.012, 0.31, 0.13, 0.1] for [A..E,G]
    # (P is whatever closes the balance; the closed-loop run lands mp ~ 0.083).
    # Sanity check: r1 should be on the order of FA=1.8275 kg/s.
    MR = par.massT
    xa, xb, xc, xp = 0.37, 0.012, 0.31, 0.083
    r1mr, r2mr, r3mr = mass_rates(xa, xb, xc, xp, MR, Tr0, par)
    print("\nAt the upstream steady state (rough mass-fraction estimate):")
    print(f"  r1mr = {r1mr:.4f} kg/s  (FA = {par.FA})")
    print(f"  r2mr = {r2mr:.4f} kg/s")
    print(f"  r3mr = {r3mr:.4f} kg/s")
    print(f"  Qrmr = {reaction_heat(r1mr, r2mr, r3mr, par):.4f} kW")
