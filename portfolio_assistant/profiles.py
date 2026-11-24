"""Client risk profiles: WHO a strategy is judged for, fixed before any test is run.

A strategy is not good or bad in itself: it is good or bad for a given client. Each
profile sets three things, all written here BEFORE looking at any result:

1. a reference portfolio, the simple allocation an adviser would propose to that client
   without any strategy: a share of equities (equal weights among the equity assets of
   the universe), the rest in bonds (equal weights among the bond assets) or, where the
   wrapper has none (PEA), in cash. Rebalanced like the strategy, same fees.
2. the judge: the client's risk aversion γ in the certainty equivalent. The higher γ,
   the more a loss weighs against a gain of the same size.
3. hard limits: a strategy whose volatility or worst loss exceeds the profile's limits
   is UNSUITABLE for that client, whatever its return.

The figures are conventions, in line with common French advisory practice: three
profiles, volatility ceilings placed in the PRIIPs risk-indicator bands (SRI 3 covers
5-12 % volatility, SRI 4 covers 12-20 %). They are deliberately simple and stated once.
Changing them after seeing results would be choosing the judge that makes a strategy
win: the lab records the profile with every trial, freeze and test.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .strategies import fixed_weights


@dataclass(frozen=True)
class Profile:
    name: str
    label: str
    gamma: float            # risk aversion of the judge (certainty equivalent)
    equity_share: float     # equities in the reference portfolio
    max_volatility: float   # above this annualised volatility: unsuitable
    max_loss: float         # worse than this drawdown: unsuitable (positive number)

    def to_dict(self) -> dict:
        return asdict(self)


PROFILES: dict[str, Profile] = {
    "prudent": Profile("prudent", "Prudent", gamma=6.0, equity_share=0.25,
                       max_volatility=0.08, max_loss=0.15),
    "equilibre": Profile("equilibre", "Équilibré", gamma=3.0, equity_share=0.50,
                         max_volatility=0.12, max_loss=0.25),
    "dynamique": Profile("dynamique", "Dynamique", gamma=1.5, equity_share=0.80,
                         max_volatility=0.20, max_loss=0.40),
}


def get(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"profil inconnu « {name} » ; profils : {', '.join(PROFILES)}") from None


def reference_weights(profile: Profile, universe: list[dict]) -> dict[str, float]:
    """The profile's reference portfolio on a universe of {"ticker", "classe"}.

    Equities: every class whose name starts with "actions". Defensive part: the
    "obligations" class; if the universe has none, that part stays in cash."""
    equities = [u["ticker"].upper() for u in universe if str(u.get("classe", "")).startswith("actions")]
    bonds = [u["ticker"].upper() for u in universe if u.get("classe") == "obligations"]
    if not equities:
        raise ValueError("the universe has no equity asset: no reference portfolio for a profile")
    weights = {t: profile.equity_share / len(equities) for t in equities}
    for t in bonds:
        weights[t] = (1 - profile.equity_share) / len(bonds)
    return weights


def reference_rule(profile: Profile, universe: list[dict]):
    """Factory of the reference rule (a fixed allocation, rebalanced)."""
    weights = reference_weights(profile, universe)
    return lambda: fixed_weights(weights)


def suitability(profile: Profile, volatility: float, max_drawdown: float) -> list[str]:
    """The profile limits a result breaks (empty list = suitable)."""
    problems = []
    if volatility > profile.max_volatility:
        problems.append(f"volatilité {volatility:.1%} > {profile.max_volatility:.0%} autorisés")
    if max_drawdown < -profile.max_loss:
        problems.append(f"pire chute {max_drawdown:.1%} au-delà de -{profile.max_loss:.0%} tolérés")
    return problems
