from az.player_tactical import TacticalAZPlayer
from az.threat_space import ThreatSpaceAZPlayer


def wrap_az_player(base_player, cfg: dict | None = None):
    """Apply configured tactical wrappers to an already-built AZ player."""
    cfg = cfg or {}
    tactical = cfg.get("tactical") or {}
    threat = cfg.get("threat_space") or {}

    if threat.get("enabled"):
        return ThreatSpaceAZPlayer(base_player, tactical_enabled=bool(tactical.get("enabled")), cfg=cfg)
    if tactical.get("enabled"):
        return TacticalAZPlayer(base_player, cfg=cfg)
    return base_player
