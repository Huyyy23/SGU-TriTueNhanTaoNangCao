from az.player_factory import wrap_az_player
from az.player_tactical import TacticalAZPlayer
from az.threat_space import ThreatSpaceAZPlayer


class BasePlayer:
    rules = object()


def test_wrap_az_player_returns_base_when_overrides_disabled():
    base = BasePlayer()

    assert wrap_az_player(base, {}) is base


def test_wrap_az_player_uses_tactical_wrapper_when_enabled():
    player = wrap_az_player(BasePlayer(), {"tactical": {"enabled": True}})

    assert isinstance(player, TacticalAZPlayer)
    assert not isinstance(player, ThreatSpaceAZPlayer)


def test_wrap_az_player_uses_threat_wrapper_and_keeps_tactical_enabled():
    player = wrap_az_player(
        BasePlayer(),
        {"tactical": {"enabled": True}, "threat_space": {"enabled": True}},
    )

    assert isinstance(player, ThreatSpaceAZPlayer)
    assert player.enabled
    assert player.threat_enabled
