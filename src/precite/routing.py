"""Politique de routing temporel pour les modeles de prediction.

Le routing determine quel modele (annuel ou production) utiliser pour
predire les citations d'un article publie une annee donnee.

Regle :
1. Si target_year <= last_complete_year :
   - utiliser le modele annuel de target_year s'il existe (route "historical")
   - sinon le modele annuel de l'annee la plus proche <= target_year
     (route "fallback_before")
   - sinon le modele annuel le plus ancien (route "fallback_early")
2. Sinon (target_year > last_complete_year) :
   - utiliser le modele production le plus recent (route "production")
   - si pas de production, fallback sur le dernier annuel
     (route "fallback_production")
3. Aucun modele disponible -> route "none", year=None

Cette politique est utilisee par :
- batch.py (ecriture) : selectionne le modele MLflow a charger
- parquet_store.py (lecture) : selectionne le fichier Parquet a lire

Les deux DOIVENT rester synchronisees. Toute modification doit passer
par ce module et etre testee par tests/unit/test_model_selection.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingDecision:
    """Resultat du routing : quelle annee de modele utiliser, et pourquoi."""

    year: int | None
    """Annee du modele selectionne (None si aucun modele disponible)."""

    route: str
    """Raison du choix : historical, production, fallback_early,
    fallback_before, fallback_production, none."""

    is_production: bool
    """True si le modele selectionne est un modele production."""


def select_routing(
    target_year: int,
    *,
    annual_years: set[int] | list[int] | None,
    production_years: set[int] | list[int] | None,
    last_complete_year: int | None,
) -> RoutingDecision:
    """Selectionne l'annee de modele a utiliser pour un target_year donne.

    Args:
        target_year: annee de publication de l'article a predire.
        annual_years: annees des modeles annuels disponibles.
        production_years: annees des modeles production disponibles.
        last_complete_year: derniere annee avec cible complete.

    Returns:
        RoutingDecision avec l'annee selectionnee et la route.
    """
    annuals = set(annual_years or [])
    productions = set(production_years or [])

    # ─── Cas 1 : target_year <= last_complete_year -> modele annuel ───
    if last_complete_year is None or target_year <= last_complete_year:
        # Exact match
        if target_year in annuals:
            return RoutingDecision(
                year=target_year, route="historical", is_production=False,
            )

        # Plus proche <= target_year
        candidates = [y for y in annuals if y <= target_year]
        if candidates:
            return RoutingDecision(
                year=max(candidates), route="fallback_before", is_production=False,
            )

        # Fallback : le plus ancien disponible
        if annuals:
            return RoutingDecision(
                year=min(annuals), route="fallback_early", is_production=False,
            )

    # ─── Cas 2 : target_year > last_complete_year -> production ──────
    if productions:
        return RoutingDecision(
            year=max(productions), route="production", is_production=True,
        )

    # ─── Fallback : dernier annuel si pas de production ───────────────
    if annuals:
        return RoutingDecision(
            year=max(annuals), route="fallback_production", is_production=False,
        )

    return RoutingDecision(year=None, route="none", is_production=False)