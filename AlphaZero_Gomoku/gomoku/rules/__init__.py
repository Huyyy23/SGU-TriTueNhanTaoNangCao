"""Rule sets. Callers use only the Rules interface."""

from gomoku.rules.base import Rules
from gomoku.rules.normal import NormalRules
from gomoku.rules.renju import RenjuRules

_RULES = {
    "normal": NormalRules,
    "renju": RenjuRules,
}


def get_rules(name: str) -> Rules:
    """Create a rule set by name."""
    try:
        rules_class = _RULES[name]
    except KeyError:
        raise ValueError(f"unknown rule set {name!r}; choose from {sorted(_RULES)}") from None
    return rules_class()


__all__ = ["Rules", "NormalRules", "RenjuRules", "get_rules"]
